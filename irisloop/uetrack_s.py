"""UETrack-S as a single-object measurer.

The official checkout lives in ``third_party/UETrack`` (or ``$UETRACK_ROOT``).
The first user box is the template and stays frozen. Later frames are matched
only inside a search crop around the previous box. RGB frames are repeated to
the 6-channel input the checkpoint expects, and an empty language token is
encoded once because the published Small model was built with that path on.

This module does not import torch until a model is constructed, so the rest of
IrisLoop runs without it.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from irisloop.single_object import BBox, BoxMeasurement


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_ROOT = os.path.join(_ROOT, "third_party", "UETrack")
_CHECKPOINT_URL = (
    "https://huggingface.co/kangben258/UETrack/resolve/main/"
    "uetrack/checkpoints/uetrack_small.tar"
)


def uetrack_root() -> str:
    return os.environ.get("UETRACK_ROOT") or _DEFAULT_ROOT


def default_checkpoint(root: str | None = None) -> str:
    base = root or uetrack_root()
    for name in ("uetrack_small.pth.tar", "uetrack_small.tar"):
        path = os.path.join(base, "checkpoints", name)
        if os.path.isfile(path):
            return path
    return os.path.join(base, "checkpoints", "uetrack_small.pth.tar")


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.ndim not in (2, 3):
        raise ValueError("frame must be a BGR or grayscale image")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_uetrack(root: str):
    if not os.path.isfile(os.path.join(root, "lib", "models", "uetrack", "uetrack.py")):
        raise RuntimeError(
            f"UETrack source not found at {root}. "
            "Clone https://github.com/kangben258/UETrack into third_party/UETrack "
            "or set UETRACK_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "UETrack-S needs PyTorch. On this machine:\n"
            "  python -m pip install torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  python -m pip install timm==0.9.16 einops easydict pyyaml "
            "git+https://github.com/openai/CLIP.git"
        ) from exc
    import lib.models.uetrack.encoder as encoder_module

    # The factory downloads a backbone whenever this returns true. The
    # checkpoint replaces those weights, so skip the download.
    encoder_module.is_main_process = lambda: False
    from lib.config.uetrack.config import cfg, update_config_from_file
    from lib.models.uetrack import build_uetrack_inference
    from lib.test.tracker.utils import sample_target, transform_image_to_crop
    from lib.test.utils.hann import hann2d
    from lib.utils.box_ops import clip_box

    return {
        "torch": torch,
        "cfg": cfg,
        "update_config": update_config_from_file,
        "build": build_uetrack_inference,
        "sample_target": sample_target,
        "transform_image_to_crop": transform_image_to_crop,
        "hann2d": hann2d,
        "clip_box": clip_box,
    }


class UETrackS:
    """Frozen-template UETrack-S. ``start`` / ``track`` return a box and a score."""

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        root: str | None = None,
        device: str | None = None,
    ) -> None:
        self.root = root or uetrack_root()
        self.checkpoint = checkpoint or default_checkpoint(self.root)
        if not os.path.isfile(self.checkpoint):
            raise RuntimeError(
                f"missing UETrack-S weights: {self.checkpoint}\n"
                f"Download {_CHECKPOINT_URL}"
            )
        api = _load_uetrack(self.root)
        torch = api["torch"]
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._keep_clip_on_cpu()
        cfg = api["cfg"]
        api["update_config"](os.path.join(
            self.root, "experiments", "uetrack", "uetrack_small.yaml",
        ))
        network = api["build"](cfg)
        state = self._read_checkpoint(torch)
        missing, unexpected = network.load_state_dict(state, strict=False)
        encoder_missing = [key for key in missing if key.startswith("encoder.")]
        if len(encoder_missing) > 10:
            raise RuntimeError(
                f"checkpoint {self.checkpoint} does not match UETrack-S "
                f"({len(encoder_missing)} encoder keys missing, "
                f"{len(unexpected)} unexpected)"
            )
        network.to(self.device)
        network.eval()
        self.network = network
        self.cfg = cfg
        self.sample_target = api["sample_target"]
        self.transform_image_to_crop = api["transform_image_to_crop"]
        self.clip_box = api["clip_box"]
        self.template_size = int(cfg.TEST.TEMPLATE_SIZE)
        self.search_size = int(cfg.TEST.SEARCH_SIZE)
        self.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
        self.search_factor = float(cfg.TEST.SEARCH_FACTOR)
        self.multi_modal_vision = bool(cfg.TEST.MULTI_MODAL_VISION.DEFAULT)
        self.multi_modal_language = bool(cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT)
        stride = int(cfg.MODEL.ENCODER.STRIDE)
        feat = self.search_size // stride
        window = api["hann2d"](torch.tensor([feat, feat]).long(), centered=True)
        self.output_window = window.to(self.device) if bool(cfg.TEST.WINDOW) else None
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.task_index = torch.tensor([0], device=self.device)
        self.state: list[float] | None = None
        self._previous: list[float] | None = None
        self.template_list = None
        self.template_anno_list = None
        self.text_src = None

    @staticmethod
    def _keep_clip_on_cpu() -> None:
        """The published builder loads CLIP onto CUDA during construction.

        The rest of the network is still on CPU at that moment. Keep CLIP on
        CPU and move the whole model together after the checkpoint is loaded.
        """
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError(
                "UETrack-S needs the CLIP text tower. Install it with:\n"
                "  python -m pip install git+https://github.com/openai/CLIP.git"
            ) from exc
        original = clip.load

        def load_on_cpu(name, device=None, jit=False, download_root=None):
            del device
            return original(name, device="cpu", jit=jit, download_root=download_root)

        clip.load = load_on_cpu

    @staticmethod
    def _allow_training_checkpoint_imports() -> None:
        """The published weights pickle a few training-only objects.

        Unpickling them imports ``lib.train``. That package init pulls in a
        TensorBoard writer and a machine-local path file. Inference only
        needs the ``net`` state dict, so those imports are satisfied with
        empty stand-ins.
        """
        import types

        if "tensorboardX" not in sys.modules:
            stub = types.ModuleType("tensorboardX")

            class SummaryWriter:
                def __init__(self, *args, **kwargs) -> None:
                    del args, kwargs

            stub.SummaryWriter = SummaryWriter
            sys.modules["tensorboardX"] = stub
        import lib.train.admin.stats  # noqa: F401  runs the package init

        name = "lib.train.admin.local"
        if name not in sys.modules:
            local = types.ModuleType(name)

            class EnvironmentSettings:
                pass

            local.EnvironmentSettings = EnvironmentSettings
            sys.modules[name] = local

    def _read_checkpoint(self, torch) -> dict:
        self._allow_training_checkpoint_imports()
        try:
            blob = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(self.checkpoint, map_location="cpu")
        state = blob["net"] if isinstance(blob, dict) and "net" in blob else blob
        if not isinstance(state, dict):
            raise RuntimeError(f"{self.checkpoint} has no state dict")
        if any(key.startswith("module.") for key in state):
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
        return state

    def reset(self) -> None:
        self.state = None
        self._previous = None
        self.template_list = None
        self.template_anno_list = None
        self.text_src = None

    def _preprocess(self, patch: np.ndarray):
        tensor = self.torch.from_numpy(np.ascontiguousarray(patch))
        tensor = tensor.to(self.device).float().permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor / 255.0 - self.mean) / self.std
        if self.multi_modal_vision and tensor.size(1) == 3:
            tensor = self.torch.cat((tensor, tensor), dim=1)
        return tensor

    def _encode_text(self) -> None:
        if not self.multi_modal_language:
            self.text_src = None
            return
        tokens = self.torch.zeros(1, 77, dtype=self.torch.long, device=self.device)
        with self.torch.inference_mode():
            self.text_src = self.network.forward_textencoder_inference(tokens)

    def start(self, frame: np.ndarray, bbox: BBox) -> BoxMeasurement:
        self.reset()
        x, y, width, height = (float(v) for v in bbox)
        if min(width, height) < 8 or not np.isfinite([x, y, width, height]).all():
            raise ValueError("draw a larger box")
        image = _as_rgb(frame)
        patch, resize = self.sample_target(
            image, [x, y, width, height], self.template_factor,
            output_sz=self.template_size,
        )
        template = self._preprocess(patch)
        torch = self.torch
        anno = self.transform_image_to_crop(
            torch.tensor([x, y, width, height]),
            torch.tensor([x, y, width, height]),
            resize,
            torch.tensor([self.template_size, self.template_size]),
            normalize=True,
        )
        self.template_list = [template]
        self.template_anno_list = [anno.to(self.device).unsqueeze(0)]
        self.state = [x, y, width, height]
        self._encode_text()
        return BoxMeasurement((x, y, width, height), 1.0)

    def track(self, frame: np.ndarray) -> BoxMeasurement:
        if self.state is None or self.template_list is None or self.template_anno_list is None:
            raise RuntimeError("no active selection; draw a box")
        self._previous = list(self.state)
        image = _as_rgb(frame)
        patch, resize = self.sample_target(
            image, self.state, self.search_factor, output_sz=self.search_size,
        )
        search = self._preprocess(patch)
        torch = self.torch
        with torch.inference_mode():
            encoded, _, _ = self.network.forward_encoder(
                self.template_list, [search], self.template_anno_list,
                self.text_src, self.task_index,
            )
            out = self.network.forward_decoder(feature=encoded)
        score_map = out["score_map"]
        response = score_map if self.output_window is None else self.output_window * score_map
        if "size_map" in out:
            pred_boxes, conf = self.network.decoder.cal_bbox(
                response, out["size_map"], out["offset_map"], return_score=True,
            )
        else:
            pred_boxes, conf = self.network.decoder.cal_bbox(
                response, out["offset_map"], return_score=True,
            )
        pred = (pred_boxes.view(-1, 4).mean(dim=0) * self.search_size / resize).tolist()
        height, width = image.shape[:2]
        self.state = self.clip_box(self._map_box_back(pred, resize), height, width, margin=10)
        score = float(conf.detach().float().reshape(-1)[0].item())
        box = tuple(float(v) for v in self.state)
        return BoxMeasurement(box, score)  # type: ignore[arg-type]

    def _map_box_back(self, pred_box: list[float], resize: float) -> list[float]:
        """Place a crop-relative box back into the full image.

        ``pred_box`` is ``(cx, cy, w, h)`` in image pixels, measured from the
        top-left of the search crop. This is the official UETrack mapping.
        """
        assert self.state is not None
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, width, height = pred_box
        half = 0.5 * self.search_size / resize
        cx_real = cx + (cx_prev - half)
        cy_real = cy + (cy_prev - half)
        return [cx_real - 0.5 * width, cy_real - 0.5 * height, width, height]

    def revert(self) -> None:
        """Drop the last box so a rejected score does not move the search."""
        if self._previous is not None:
            self.state = self._previous
