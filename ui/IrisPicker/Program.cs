using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace IrisPicker
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            bool created;
            using (var mutex = new Mutex(true, "IrisLoop.IrisPicker", out created))
            {
                if (!created)
                {
                    ActivateExisting();
                    return;
                }
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                try
                {
                    Application.Run(new PickerForm());
                }
                catch (Exception ex)
                {
                    var log = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "startup.log");
                    File.WriteAllText(log, ex.ToString());
                    MessageBox.Show(ex.Message, "IrisPicker 无法启动", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private static void ActivateExisting()
        {
            var handle = FindWindow(null, "选择 IrisGreen");
            if (handle == IntPtr.Zero)
                return;
            ShowWindow(handle, 9);
            SetForegroundWindow(handle);
        }

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr FindWindow(string className, string windowName);

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr handle);

        [DllImport("user32.dll")]
        private static extern bool ShowWindow(IntPtr handle, int command);
    }

    internal sealed class PickerForm : Form
    {
        private readonly ListView _devices;
        private readonly Label _chosen;
        private readonly Button _scan;
        private readonly Button _project;
        private readonly TextBox _log;
        private readonly string _repo;
        private readonly string _python;
        private string _chosenName;
        private string _chosenAddress;
        private bool _busy;

        internal PickerForm()
        {
            Text = "选择 IrisGreen";
            Font = new Font("Microsoft YaHei UI", 9F);
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(760, 480);
            MinimumSize = new Size(640, 420);

            _repo = FindRepo();
            _python = FindPython(_repo);

            var hint = new Label
            {
                Dock = DockStyle.Top,
                Height = 36,
                Padding = new Padding(12, 10, 12, 0),
                Text = "扫描只列出正在广播的 IrisGreen。有多台时必须点选一台，不会自动用列表里的第一台。",
            };

            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 44,
                Padding = new Padding(12, 8, 12, 0),
                FlowDirection = FlowDirection.LeftToRight,
            };
            _scan = new Button { Text = "扫描", Width = 88, Height = 28 };
            _project = new Button { Text = "投射到所选设备", Width = 140, Height = 28, Enabled = false };
            _scan.Click += async (sender, args) => await ScanAsync();
            _project.Click += async (sender, args) => await ProjectAsync();
            buttons.Controls.Add(_scan);
            buttons.Controls.Add(_project);

            _chosen = new Label
            {
                Dock = DockStyle.Top,
                Height = 28,
                Padding = new Padding(12, 4, 12, 0),
                Text = "已选择：无",
            };

            _log = new TextBox
            {
                Dock = DockStyle.Bottom,
                Height = 72,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
            };

            _devices = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true,
                GridLines = true,
                HideSelection = false,
                MultiSelect = false,
            };
            _devices.Columns.Add("名称", 220);
            _devices.Columns.Add("地址", 220);
            _devices.Columns.Add("信号 (dBm)", 100);
            _devices.SelectedIndexChanged += (sender, args) => UpdateChoice();

            Controls.Add(_devices);
            Controls.Add(_log);
            Controls.Add(_chosen);
            Controls.Add(buttons);
            Controls.Add(hint);

            _log.Text = "Python  " + _python;
        }

        private void UpdateChoice()
        {
            if (_busy)
                return;
            if (_devices.SelectedItems.Count == 0)
            {
                _chosenName = null;
                _chosenAddress = null;
                _chosen.Text = "已选择：无";
                _project.Enabled = false;
                return;
            }
            var item = _devices.SelectedItems[0];
            _chosenName = item.Text;
            _chosenAddress = Convert.ToString(item.Tag);
            _chosen.Text = "已选择：" + _chosenName + "    " + _chosenAddress;
            _project.Enabled = true;
        }

        private async Task ScanAsync()
        {
            SetBusy(true, "正在扫描…");
            try
            {
                var output = await Task.Run(() => RunSession("find --seconds 8"));
                var parsed = new JavaScriptSerializer().Deserialize<System.Collections.Generic.Dictionary<string, object>>(output);
                if (parsed != null && parsed.ContainsKey("error") && parsed["error"] != null)
                    throw new InvalidOperationException(Convert.ToString(parsed["error"]));

                _devices.Items.Clear();
                var devices = parsed == null ? null : parsed["devices"] as System.Collections.ArrayList;
                if (devices != null)
                {
                    foreach (var entry in devices)
                    {
                        var device = entry as System.Collections.Generic.Dictionary<string, object>;
                        if (device == null)
                            continue;
                        var name = Convert.ToString(device["name"]);
                        var address = Convert.ToString(device["address"]);
                        var rssi = device["rssi"] == null ? "" : Convert.ToString(device["rssi"]);
                        var item = new ListViewItem(name);
                        item.SubItems.Add(address);
                        item.SubItems.Add(rssi);
                        item.Tag = address;
                        _devices.Items.Add(item);
                    }
                }
                _log.Text = _devices.Items.Count == 0
                    ? "没有扫到 IrisGreen。设备需要正在广播；这台电脑的 WLAN 若和蓝牙共用网卡，连着 Wi-Fi 时会漏扫。"
                    : "扫到 " + _devices.Items.Count + " 台。点一行再投射。";
            }
            catch (Exception ex)
            {
                _log.Text = ex.Message;
            }
            finally
            {
                SetBusy(false, null);
                UpdateChoice();
            }
        }

        private async Task ProjectAsync()
        {
            if (string.IsNullOrEmpty(_chosenAddress))
                return;
            var name = _chosenName;
            var address = _chosenAddress;
            var confirm = MessageBox.Show(
                "向 " + name + "\n" + address + "\n投射 8 帧测试动画。这会替换第 1 组开机素材。",
                "确认所选设备",
                MessageBoxButtons.OKCancel,
                MessageBoxIcon.Warning);
            if (confirm != DialogResult.OK)
                return;

            SetBusy(true, "正在向 " + name + " 投射…");
            _chosen.Text = "已选择：" + name + "    " + address;
            try
            {
                var output = await Task.Run(() => RunSession("project --address " + address + " --seconds 60 --frame-ms 300"));
                var parsed = new JavaScriptSerializer().Deserialize<System.Collections.Generic.Dictionary<string, object>>(output);
                if (parsed != null && parsed.ContainsKey("error") && parsed["error"] != null)
                    throw new InvalidOperationException(Convert.ToString(parsed["error"]));
                var replaced = parsed == null ? "" : Convert.ToString(parsed["replaced"]);
                var seconds = parsed == null ? "" : Convert.ToString(parsed["play_seconds"]);
                _log.Text = "已投到 " + address + "，" + replaced + "，播放 " + seconds + " 秒。";
            }
            catch (Exception ex)
            {
                _log.Text = ex.Message;
            }
            finally
            {
                SetBusy(false, null);
                UpdateChoice();
            }
        }

        private void SetBusy(bool busy, string status)
        {
            _busy = busy;
            _scan.Enabled = !busy;
            _project.Enabled = !busy && _devices.SelectedItems.Count == 1;
            if (status != null)
                _log.Text = status;
        }

        private string RunSession(string arguments)
        {
            var start = new ProcessStartInfo
            {
                FileName = _python,
                Arguments = "-m irisloop.session " + arguments,
                WorkingDirectory = _repo,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            using (var process = Process.Start(start))
            {
                var stdout = process.StandardOutput.ReadToEnd();
                var stderr = process.StandardError.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    var message = string.IsNullOrWhiteSpace(stdout) ? stderr : stdout;
                    throw new InvalidOperationException(string.IsNullOrWhiteSpace(message)
                        ? "irisloop.session 退出码 " + process.ExitCode
                        : message.Trim());
                }
                return stdout;
            }
        }

        private static string FindRepo()
        {
            var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
            while (dir != null)
            {
                var session = Path.Combine(dir.FullName, "irisloop", "session.py");
                if (File.Exists(session))
                    return dir.FullName;
                dir = dir.Parent;
            }
            throw new InvalidOperationException("找不到 IrisLoop 仓库（需要 irisloop/session.py）。");
        }

        private static string FindPython(string repo)
        {
            var venv = Path.Combine(repo, ".venv", "Scripts", "python.exe");
            if (File.Exists(venv))
                return venv;
            return "python";
        }
    }

}
