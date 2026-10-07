# For-A：反差萌 JK 手办生成器

把本人的脸做成 3D 头像，装到一个漂亮的 JK 制服美少女身体上，输出可直接在拓竹 A1 mini 上打印的 STL。

## 成品（`out/` 目录，每个模型一个子目录）

| 模型 | 文件 | 高度 | 说明 |
|---|---|---|---|
| 全身版 | `out/his_jk_full/his_jk_full.stl` | 约 170 mm | 修长身材，害羞比心站姿，白衬衫、领带、黑百褶裙，配本人头像（寸头、黑框眼镜、闭嘴浅笑） |
| Q 版 · 秃头 | `out/his_jk_q_bald/his_jk_q_bald.stl` | 约 150 mm | 大头娃娃比例，光头，头顶一撮很多根的短毛 |
| 半身像 | `out/his_jk_bust/his_jk_bust.stl` | 约 170 mm | 从裙摆以上，脸做得最大最细，裙摆当底座 |

每个子目录里还有：

- `parts/`：多色打印分件，各自封闭、互不重叠，分为 `skin_head`（头和脸）、`body`（身体和衣服）、`hair`、`glasses`、`base`。
- `report.json`：尺寸、体积、预计耗材、悬垂面积、是否水密。
- `preview_colour.png`、`preview_face.png`、`preview_single_colour.png`：渲染预览。

## 拓竹 Bambu Studio 打印设置（A1 mini，0.4 mm 喷嘴）

1. **导入模型**
   - 单色打印：导入 `<模型>.stl`。
   - 多色打印：一次性选中 `parts/` 里的全部 STL 拖进来，在弹窗里选择"作为一个对象的多个部件加载"。
2. **摆放**
   - 文件已经是底座朝下，Z 轴向上，不用旋转。
   - 所有模型都小于 180 mm，能放进 A1 mini。
3. **工艺参数**

   | 项 | 推荐值 |
   |---|---|
   | 层高 | 0.08 mm（Extra Fine） |
   | 墙 | 3 圈 |
   | 填充 | 15%，gyroid |
   | 支撑 | 树状（Tree, auto） |
   | 支撑阈值角 | 30°–35° |
   | 支撑顶部 Z 距离 | 0.1 mm |
   | 支撑 | 开启"仅在构建板上" |
   | 外墙速度 | 降到 50–60 mm/s，脸部细节更清楚 |
   | 底座 | 不需要 brim（底座本身就是圆盘） |

4. **需要支撑的地方**
   - 下巴、眼镜腿、比心的手和小臂下方、裙摆下沿。
   - 鞋跟和底座之间无需支撑。
5. **耗材**
   - 推荐哑光 PLA，脸部不反光，更像手办。
   - 多色方案（AMS lite 4 色）：
     - 肤色：`skin_head`、`body`
     - 黑色：`hair`、`glasses`
     - 白或灰：`base`
   - 如果想让衣服分色（白衬衫、黑裙），可以在 Bambu Studio 里用"上色"工具直接涂 `body`。

## 怎么重新生成

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python numpy scipy scikit-image trimesh opencv-python-headless mediapipe \
    matplotlib fast-simplification rtree networkx pillow libigl open3d pygltflib manifold3d pytest
PYTHONPATH=. .venv/bin/python -m figurine.build full        # 或 chibi_bald / chibi_long / bust
PYTHONPATH=. .venv/bin/python -m figurine.finalize          # 生成彩色预览
PYTHONPATH=. .venv/bin/python -m pytest tests
```

需要的私有素材都放在 `assets/`（已被 `.gitignore` 忽略，不会上传）：

- `assets/photos/`：本人照片。
- `assets/models/`：MediaPipe 模型，即 `face_landmarker.task`、`canonical_face_model.obj`、`selfie_multiclass.tflite`。
- `assets/ref_models/beauty/luntima_student/`：身体模型。

## 代码结构

| 模块 | 作用 |
|---|---|
| `figurine/face.py` | 多张照片 → MediaPipe 478 个关键点 → 转正、融合 |
| `figurine/head2.py` | 本人头像：关键点网格细分成光滑脸，拟合头型，再加上美化、立体眼睛（眼球、虹膜、瞳孔、高光）、薄唇浅笑、M 形发际线寸头、外张的耳朵、黑框眼镜，以及 Q 版头顶短毛（`crown_tufts`） |
| `figurine/assemble.py` | 在脖子处切掉原模型的头，按下巴和脖子对齐装上本人的头 |
| `figurine/build.py` | 生成四个成品和多色分件 |
| `figurine/sdf.py`、`mesher.py`、`meshsdf.py` | 距离场建模、任意网格转实体、提取水密网格 |
| `figurine/check.py`、`preview.py`、`colorize.py` | 打印检查、无 GPU 渲染、贴图取色预览 |

## 素材来源与许可

- **身体模型：** This work is based on "Fanart Garin BlackX - Luntima student girl" (https://sketchfab.com/3d-models/fanart-garin-blackx-luntima-student-girl-ca7d9222a2864e2897e640171e22a24f) by Seraphim Sarov (https://sketchfab.com/Songti) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)。原角色为 Garin BlackX 的 Luntima 同人。头部、头发、眼睛已全部替换。
- **人脸关键点：** MediaPipe Face Landmarker 和 canonical face model，Apache-2.0。
- **本人照片：** 仅在本地处理，不上传、不入库。
