# IrisGreen 蓝牙连不上的情况

IrisGreen（广播名形如 `Iris-G-XXXXXX`）靠 BLE 广播才能被发现，靠一次完整握手才能连上。下面是仓库里已经踩过、并写进代码或现场记录的情况。分两段：扫不到，和扫到了但握手失败或随后卡死。

地址会随机器变。现场用过的两台：

| 机器 | 地址 |
|---|---|
| 旧机 | `F4:12:FA:B6:B7:CA` |
| 新机 | `20:6E:F1:81:71:62` |

`find` 扫到几台就列几台。零台和多台都不自动挑一台。

## 扫不到

`python -m irisloop.session find` 的 `devices` 为空，或界面写「没有扫到 IrisGreen」。这时还没有进入连接。

| 情况 | 表现 | 记录位置 |
|---|---|---|
| 休眠或没电，设备不广播 | 2026-10-10 新机 `20:6E:F1:81:71:62` 完全不出现在扫描结果里 | `experience/iteration-log/2026-10-10-procalib-inverse-camera.md` |
| 这台电脑的 WLAN 和蓝牙共用网卡，Wi-Fi 正连着 | 设备开着也漏扫 | `ui/IrisPicker/Program.cs` 扫描为空时的提示 |
| 设备停在自身 Wi-Fi AP，没有切回 BLE | 拉图流程会关掉 BLE：写 `adb40005` 的 `0x02` 打开热点。不请求 `http://192.168.4.1/switch/ble` 就不会回到可连接的广播 | `tools/pull_images.py` |

没指定地址、且一台都没扫到时，`resolve_address` 报 `no IrisGreen is advertising`。

## 扫到了，但连不上或连上就卡死

| 情况 | 表现 | 代码怎么处理 |
|---|---|---|
| 握手被中途打断 | 固件楔住。立刻重试只会让恢复更慢 | `session.connect` 失败即抛出，不重试。`BleakClient` 超时 60 秒，避免在握手中途放弃。失败时尝试断一次，断不开也只吞掉这次异常 |
| 同一进程里断开再连 | 固件同样会楔住。Windows 上 Bleak 重连也会失败 | `tools/aim_at.py` 整个进程只连一次，镜头之间不重连。`tools/capture_vu_testset.py` 每一组放进独立子进程，组与组之间留时间让 BLE 栈冷却 |
| 扫描刚结束就连接 | 协议栈还没让出扫描，连接挂起 | `tools/aim_at.py` 在 `connect` 前先等 1 秒 |
| Windows 上只拿 MAC 字符串去连，不用刚扫到的设备对象 | WinRT 不能马上连上 | `tools/pull_images.py` 的 `find_device` 把扫描得到的 `BLEDevice` 交给 `BleakClient` |
| 附近不止一台，又没给地址 | 不会连；也不会偷偷用列表第一台 | `resolve_address` 报 `more than one IrisGreen is nearby; pass --address`。IrisPicker 必须点选一行 |
| 地址是另一台机器的，或这台已经不广播 | 按字符串去连会超时 | 先 `find`，再用扫到的那条地址 |

`session.close` 断链失败只打 `[warn] BLE disconnect failed`，不把异常再抛出去。这条链路若半开着，下一次连接会落到上面的「握手被打断」或「同进程重连」。

## 广播正常，但不要当成能用

2026-10-10 旧机 `F4:12:FA:B6:B7:CA` 仍在广播，GATT 也能读，`get_brightness` 和 `battery_raw` 都是 100%，镜头没有绿光。电量和亮度读数不能用来判断这台设备有没有电。先看镜头是否出光，再开始推图。
