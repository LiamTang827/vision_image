# 🚀 加速版实例搜索系统

## 关键优化点

### 1. **离线索引 + 在线查询架构**
   - **离线阶段**：一次性提取所有28,493张图库图片的特征，保存为 `.npz` 文件
   - **在线阶段**：加载索引，在内存中快速比较，无需重复计算

### 2. **GPU批处理加速**
   - 使用 `torch.utils.data.DataLoader`
   - `batch_size=128`：一次处理128张图片
   - `num_workers=8`：8个线程并行加载数据
   - `pin_memory=True`：加速GPU数据传输
   - 充分利用RTX 3070的计算能力

### 3. **向量化相似度计算**
   - 原来：循环计算每张图片的相似度
   - 现在：使用矩阵乘法一次性计算所有相似度
   - `similarities = query_features @ gallery_features.T`

### 4. **并行I/O**
   - Method 1 (ResNet)：GPU批处理 + DataLoader多线程
   - Method 2 (SIFT+Histogram)：ThreadPoolExecutor多线程并行提取特征

## 使用方法

### 快速开始（推荐）

```bash
# 直接运行加速版本
python instance_search_fast.py
```

**第一次运行**：
- 会构建索引文件（约1-2分钟，取决于GPU性能）
- 生成 `gallery_index_resnet.npz` 和 `gallery_index_hybrid.npz`

**后续运行**：
- 自动加载已有索引（秒级）
- 50个查询在几秒到几十秒内完成

### 强制重建索引

```bash
# 编辑 instance_search_fast.py，将 force_rebuild 改为 True
system_resnet.build_gallery_index('gallery_index_resnet.npz', force_rebuild=True)
```

### 性能对比测试

```bash
# 运行性能对比脚本
python benchmark_speed.py
```

选择选项3查看新旧方法的速度对比。

## 预期性能提升

基于RTX 3070：

| 方法 | 旧版（串行） | 新版（并行+GPU） | 加速比 |
|------|------------|----------------|--------|
| **ResNet50索引构建** | N/A (每次都重算) | 30-60秒（一次性） | ∞ |
| **ResNet50单次查询** | 60-120秒 | 0.1-0.5秒 | **100-500x** |
| **SIFT+Histogram索引** | N/A | 60-120秒（一次性） | ∞ |
| **SIFT+Histogram查询** | 30-60秒 | 0.5-2秒 | **30-60x** |
| **完成50个查询** | 60-90分钟 | 2-5分钟 | **20-30x** |

## 技术细节

### ResNet50批处理流程

```python
# 1. 创建Dataset和DataLoader
dataset = GalleryDataset(gallery_path, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=128,      # RTX 3070可以处理大批量
    num_workers=8,       # 并行加载
    pin_memory=True,     # 加速GPU传输
    prefetch_factor=2    # 预取数据
)

# 2. 批量提取特征
for batch_imgs, batch_ids, batch_names in dataloader:
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    features = resnet_model(batch_imgs)  # GPU批处理
    # 保存特征...

# 3. 保存索引
np.savez_compressed('gallery_index.npz', 
                   features=features, 
                   ids=ids, 
                   names=names)
```

### 快速查询流程

```python
# 1. 加载预计算的索引
data = np.load('gallery_index.npz')
gallery_features = data['features']  # (28493, 2048)

# 2. 提取查询特征
query_features = extract_query_features(query_img)  # (num_rois, 2048)

# 3. 向量化相似度计算（超快！）
similarities = query_features @ gallery_features.T  # (num_rois, 28493)
max_similarities = np.max(similarities, axis=0)     # (28493,)

# 4. 排序获取结果
ranked_indices = np.argsort(-max_similarities)
```

## 文件说明

- `instance_search_fast.py` - 加速版主程序
- `instance_search.py` - 原版程序（保留作为备份）
- `benchmark_speed.py` - 性能对比工具
- `gallery_index_resnet.npz` - ResNet特征索引（自动生成）
- `gallery_index_hybrid.npz` - SIFT+Histogram索引（自动生成）
- `rankList_ResNet50.txt` - Method 1输出
- `rankList_SIFT_Histogram.txt` - Method 2输出
- `results_query_*_*.png` - 可视化结果

## 调优建议

### 针对RTX 3070（8GB显存）

```python
# 如果显存充足，可以增加batch size
batch_size=256  # 默认128，可尝试更大

# 如果显存不足，降低batch size
batch_size=64   # 或32

# CPU线程数根据处理器调整
num_workers=8   # i7/i9可用8-12，i5用4-6
```

### 内存不足怎么办

如果28,493张图的特征占用内存太大：

```python
# 使用内存映射
gallery_features = np.load('gallery_index.npz', mmap_mode='r')['features']
```

## 故障排查

### GPU未被使用
```python
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### DataLoader卡住
```python
# 降低num_workers
num_workers=0  # 单线程模式
```

### 显存不足
```python
# 降低batch_size
batch_size=32  # 或16
```

## 为什么这么快？

1. **避免重复计算**：索引只构建一次，查询时直接使用
2. **GPU并行**：128张图同时处理，比串行快100倍
3. **多线程I/O**：数据加载不阻塞GPU计算
4. **向量化操作**：NumPy矩阵运算，避免Python循环
5. **内存优化**：特征预计算并缓存在内存中

祝顺利完成作业！🎓
