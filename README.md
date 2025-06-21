# Rust-based CV 图像处理软件 实验报告

2300017751 陈思危

## 1. 项目概述

​	Rust-based CV 是一个基于 Rust 语言开发的综合图像处理软件，提供了丰富的图像操作和分析功能。该软件可作为独立工具使用，也体现出了Rust语言的种种特性，可用于Rust初学者学习。项目已上传至github：https://github.com/pkuchensw/Rust-based-CV

​	软件主要包含以下功能模块：

- **图像增强**：颜色通道增强、锐化、噪点添加
- **图像分析**：边缘检测、Harris 角点检测、特征提取
- **图像变换**：调整大小、马赛克效果、滤镜应用

​	项目采用了模块化设计，通过 GUI 界面（基于 egui 框架）提供了直观的用户交互体验，使用户能够方便地应用各种图像处理算法。项目基于Rust语言的种种优势与特点，深入体现了Rust语言相对其他语言的长处。

```rust
//! # CV Image Processing Library
//! 
//! A comprehensive image processing library that provides various image manipulation
//! and analysis functions. This library is designed to be used both as a standalone
//! tool and as part of a larger image processing pipeline.
//! 
//! ## Features
//! 
//! * Image Enhancement
//!   - Color channel manipulation
//!   - Sharpening
//!   - Noise addition
//! 
//! * Image Analysis
//!   - Edge detection
//!   - Corner detection (Harris)
//!   - Feature extraction
//! 
//! * Image Transformation
//!   - Resizing
//!   - Mosaic effect
//!   - Filtering
```

项目结构示意：
├── Cargo.toml
├── Cargo.lock
└── src
    ├── main.rs            # 主程序入口
    │   └── MyEguiApp     # GUI应用主结构体
    │       ├── new()     # 创建新应用实例
    │       ├── load_image() # 加载图像功能
    │       └── update()  # GUI更新循环
    │
    └── lib.rs            # 库文件
        ├── prints()      # 打印功能
        ├── Image Processing
        │   ├── add_noise()
        │   ├── resize_image_ratio()
        │   ├── resize_image()
        │   ├── mosaic()
        │   ├── sharpen()
        │   ├── edge_detection()
        │   ├── harris_detector()
        │   └── image_filter()
        │
        └── Color Enhancement
            ├── enhance_blue_channel()
            ├── enhance_green_channel()
            └── enhance_red_channel()

Dependencies:
├── GUI
│   ├── egui = "0.24.1"
│   ├── eframe = "0.24.1"
│   └── egui_extras = "0.24.1"
│
├── Image Processing
│   ├── image = "0.24.1"
│   └── gif = "0.11.0"
│
├── Utilities
│   ├── rand = "0.8.4"
│   ├── crossbeam = "0.8.1"
│   └── indicatif = "0.16"
│
└── System
    ├── winapi = "0.3.9"
    └── plotters = "0.3.4"

Assets:
└── src
    ├── icon.png         # 应用图标
    └── Bock-Medium-2.ttf # 自定义字体

## 2. Rust 语言特点

### 2.1 内存安全

Rust 的所有权系统是其最显著的特点之一，通过编译时检查确保内存安全，无需垃圾回收。在 CV 库中，这一特性体现在多个方面：

#### 2.1.1 图像缓冲区的处理

CV 库使用 `ImageBuffer` 类型处理图像数据，在图像处理时有以下特点

- 均使用引用传参，确保函数调用后原始图像数据仍然可用
- 避免了不必要的内存复制，直接传递原始图片的引用，提高了性能
- 编译器会静态检查确保没有其他代码同时修改这个缓冲区，防止数据竞争

```rust
/// 结果类型别名，用于简化函数签名
pub type CvResult<T> = Result<T, CvError>;
pub fn enhance_red_channel(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, factor: f32) -> CvResult<()> {
    if factor < 0.0 || factor > 2.0 {
        return Err(CvError::InvalidParameter(format!("增强因子必须在0.0-2.0之间，当前值: {}", factor)));
    }
    for (_x, _y, pixel) in image_buffer.enumerate_pixels_mut() {
        let new_r = (pixel[0] as f32 * factor).min(255.0) as u8;
        *pixel = Rgb([new_r, pixel[1], pixel[2]]);
    }
    Ok(())
}
```

#### 2.1.2 零成本抽象

Rust 的零成本抽象原则在图像处理函数中得到了充分体现，这种设计允许：

- 高级抽象不会带来运行时开销
- 编译器可以内联和优化这些函数调用
- 类型安全和内存安全在编译时得到保证

#### 2.1.3 安全的内存分配和释放

- `ImageBuffer::new` 安全地分配内存，不会导致内存泄漏
- 当` filtered_image`超出作用域时，Rust 会自动释放其内存
- 当 `*image_buffer = filtered_image` 执行时，旧的图像数据会被安全释放
- Rust 的所有权系统确保没有悬垂指针或双重释放问题

```rust
let mut filtered_image = ImageBuffer::new(width, height);
// ...
*image_buffer = filtered_image;
```

### 2.2 错误处理

Rust 的错误处理机制通过 `Result` 和 `Option` 类型提供了优雅且类型安全的错误处理方式。CV 库实现了自定义错误类型和结果类型别名：

```rust
/// 自定义错误类型，用于处理图像处理过程中可能出现的各种错误
#[derive(Debug)]
pub enum CvError {
    /// IO操作错误
    IoError(io::Error),
    /// 图像处理错误
    ImageError(image::ImageError),
    /// 参数无效错误
    InvalidParameter(String),
    /// 文件路径错误
    PathError(String),
    /// 处理过程错误
    ProcessingError(String),
}
// 从图像处理错误转换
impl From<image::ImageError> for CvError {
    fn from(err: image::ImageError) -> Self {
        CvError::ImageError(err)
    }
}
/// 结果类型别名，用于简化函数签名
pub type CvResult<T> = Result<T, CvError>;
```

错误处理代码的实际使用：

```rust
pub fn add_noise(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, degree: i16) -> CvResult<()> {
    if degree <= 0 || degree > 255 {
        return Err(CvError::InvalidParameter(format!("噪声强度必须在1-255之间，当前值: {}", degree)));
    }
    // 实现代码...
}
```

这种设计的优势：

1. 类型安全的错误处理：
   - 通过枚举区分不同类型的错误
   - 编译器强制处理所有可能的错误情况
   - 避免了传统的返回错误码或异常处理的缺点
2. 错误传播的简化：
   - 使用 `?` 运算符简化错误传播
   - 例如：`let img = image::open(&path)?;`
3. 错误转换的自动化：
   - 通过 `From` trait 实现错误类型之间的自动转换
   - 允许底层库错误无缝转换为应用级错误
4. 详细的错误信息：
   - 每种错误类型都可以携带具体的错误信息
   - 有助于调试和用户反馈

### 2.3 并发处理

Rust 的并发模型设计为"无数据竞争的并发"，这在图像处理这种计算密集型应用中尤为重要。本项目利用了 Rust 的线程安全特性：

```rust
let _semaphore = Arc::clone(&semaphore);
let handle = thread::spawn(move || {
    
    println!("Processing file: {}", path.display());
    let img = image::open(path).expect("Failed to open image");
    
    let mut img_buffer = img.to_rgb8();
    //处理图像代码部分
    print!("Saving image... thread {:?}\n" ,thread::current().id());
    img_buffer.save(output_img_path).expect("Failed to save image");
});
handles.push(handle);
```

```rust
// 等待所有线程完成
for handle in handles {
    handle.join().unwrap();
}
```

Rust 并发处理的优势：

1. 编译时线程安全检查：
   - 所有权系统确保数据在线程间安全共享
   - 编译器检测并防止数据竞争
2. 无锁并发：
   - 通过所有权转移实现线程间通信
   - 减少锁的使用，提高性能

以下是运行时间对比，在处理不同数量的图片，对比并行与串行时间效率，默认处理方式为加噪声+增强红色通道

| 图片数量 | 并行时间 | 串行时间 |
| -------- | -------- | -------- |
| 2        | 9.02s    | 27.02s   |
| 4        | 15.96s   | 53.24s   |
| 8        | 28.04s   | 104.27s  |

### 2.4 模式匹配和强大的类型系统

Rust 的模式匹配和类型系统在图像处理中提供了清晰的代码结构和类型安全：

```rust
pub fn image_filter(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, 
    filter_type: FilterType
) -> CvResult<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    match filter_type {
        FilterType::Blur(radius) => {
            Ok(image::imageops::blur(img, radius))
        },
        FilterType::Grayscale => {
            let gray_img = image::imageops::grayscale(img);
            let rgb_img = ImageBuffer::from_fn(
                gray_img.width(), 
                gray_img.height(),
                |x, y| {
                    let pixel = gray_img.get_pixel(x, y)[0];
                    Rgb([pixel, pixel, pixel])
                }
            );
            Ok(rgb_img)
        },
        FilterType::Invert => {
            let inverted = ImageBuffer::from_fn(
                img.width(), 
                img.height(),
                |x, y| {
                    let pixel = img.get_pixel(x, y);
                    Rgb([
                        255 - pixel[0],
                        255 - pixel[1],
                        255 - pixel[2]
                    ])
                }
            );
            Ok(inverted)
        },
        // 其他滤镜类型...
    }
}
pub enum FilterType {
    Blur(f32),
    Grayscale,
    Invert,
    Sepia,
    // 其他滤镜类型...
}
```

这种设计的优势：

1. 穷尽性检查：

   - 编译器确保处理了所有可能的枚举值
   - 添加新的滤镜类型时，编译器会提示更新匹配表达式

2. 代数数据类型：

   - 使用枚举表示不同的滤镜类型
   - 每种滤镜可以携带不同的参数

3. 类型安全：

   - 强类型系统确保函数参数和返回值类型正确

- 防止类型混淆和相关错误

## 3. 项目功能详解

#### 3.1 图像增强功能

CV 库提供了多种图像增强功能，包括颜色通道增强、锐化和噪点添加：

1. **颜色通道增强**：库提供了三个独立的函数来增强图像的红、绿、蓝通道，允许用户通过调整因子（0.0-2.0范围内）来控制增强程度。

```rust
pub fn enhance_red_channel(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, factor: f32) -> CvResult<()> {
    if factor < 0.0 || factor > 2.0 {
        return Err(CvError::InvalidParameter(format!("增强因子必须在0.0-2.0之间，当前值: {}", factor)));
    
    for (_x, _y, pixel) in image_buffer.enumerate_pixels_mut() {
        let new_r = (pixel[0] as f32 * factor).min(255.0) as u8;
        *pixel = Rgb([new_r, pixel[1], pixel[2]]);
    }
    Ok(())
}
```

2.**锐化处理**：通过拉普拉斯算子实现图像锐化，增强图像边缘和细节。该函数使用3×3卷积核对图像进行处理，突出显示图像中的高频成分。

```rust
pub fn sharpen(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>) {
    let lapalace_kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    let mut new_image = image.clone();
    let (width, height) = image.dimensions();
    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let mut new_pixel = [0, 0, 0];
            let mut tmp = [0, 0, 0];
            for i in 0..3 {
                for j in 0..3 {
                    let pixel = image.get_pixel(x + i - 1, y + j - 1).0;  
                    for k in 0..3 {
                        tmp[k] += (pixel[k] as i32 * lapalace_kernel[i as usize][j as usize]).max(0).min(255);
                    }
                }
            }
            for k in 0..3 {
                new_pixel[k] = tmp[k].max(0).min(255) as u8;
            }
            new_image.put_pixel(x, y, image::Rgb(new_pixel));
        }
    }
    *image = new_image;
}
```

3.**噪点添加**：为图像添加随机噪点，可用于模拟真实环境中的噪声或创建特殊效果。用户可以控制噪声强度（1-255范围内）。

```rust
pub fn add_noise(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, degree: i16) -> CvResult<()> {
    if degree <= 0 || degree > 255 {
        return Err(CvError::InvalidParameter(format!("噪声强度必须在1-255之间，当前值: {}", degree)));
    }

    let mut rng = rand::thread_rng();
    for i in 0..6 {
        for (_x, _y, pixel) in image_buffer.enumerate_pixels_mut() {
            if i % 2 == 0 {
                let new_r = (pixel[0] as i16 + rng.gen_range(1..degree)).min(255) as u8;
                let new_g = (pixel[1] as i16 + rng.gen_range(1..degree)).min(255) as u8;
                let new_b = (pixel[2] as i16 + rng.gen_range(1..degree)).min(255) as u8;
                *pixel = Rgb([new_r, new_g, new_b]);
            } else {
                let new_r = (pixel[0] as i16 - rng.gen_range(1..degree)).max(0) as u8;
                let new_g = (pixel[1] as i16 - rng.gen_range(1..degree)).max(0) as u8;
                let new_b = (pixel[2] as i16 - rng.gen_range(1..degree)).max(0) as u8;
                *pixel = Rgb([new_r, new_g, new_b]);
            }     
        }
    }
    
    Ok(())
}
```

4.**图像滤镜**：实现了一个均值滤波器，通过计算每个像素周围7×7区域的平均值来平滑图像，有效减少噪声并保留图像整体结构。

```rust
pub fn image_filter(image_buffer: &mut ImageBuffer<Rgb<u8>,Vec<u8>> ) -> CvResult<()> {
    let (width, height) = image_buffer.dimensions();  
    if width == 0 || height == 0 {
        return Err(CvError::InvalidParameter("图像尺寸不能为零".to_string()));
    }    
    let mut filtered_image = ImageBuffer::new(width, height);
    for x in 0..width {
        for y in 0..height {
            let mut sum_r = 0;
            let mut sum_g = 0;
            let mut sum_b = 0;
            let mut count = 0;
            
            for dx in -3..=3 {
                for dy in -3..=3 {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    
                    if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                        let pixel = image_buffer.get_pixel(nx as u32, ny as u32);
                        sum_r += pixel[0] as u32;
                        sum_g += pixel[1] as u32;
                        sum_b += pixel[2] as u32;
                        count += 1;
                    }
                }
            }
            
            if count == 0 {
                return Err(CvError::ProcessingError("滤镜处理时计数为零".to_string()));
            }
            let avg_r = (sum_r / count) as u8;
            let avg_g = (sum_g / count) as u8;
            let avg_b = (sum_b / count) as u8;
            filtered_image.put_pixel(x, y, Rgb([avg_r, avg_g, avg_b]));
        }
    }
    *image_buffer = filtered_image;
    Ok(())
}
```

#### 3.2 图像分析功能

CV 库实现了多种图像分析算法，包括边缘检测和角点检测：

1. **边缘检测**：使用梯度计算方法实现边缘检测，通过计算水平和垂直方向的梯度差异来识别图像中的边缘。该算法使用阈值（默认200.0）来确定是否将像素标记为边缘。

```rust
pub fn edge_detection(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>) -> CvResult<()> {
    let threshold = 200.0; // 设置阈值
    let mut new_image = image.clone();

    let (width, height) = image.dimensions();
    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let mut new_pixel = [0, 0, 0];
            let pixel1= image.get_pixel(x, y-1).0;
            let pixel2= image.get_pixel(x, y+1).0;
            let pixel3= image.get_pixel(x+1, y).0;
            let pixel4= image.get_pixel(x-1, y).0;
            let gx = (pixel3[0] as i32 - pixel4[0] as i32).abs() + 
                     (pixel3[1] as i32 - pixel4[1] as i32).abs() +
                     (pixel3[2] as i32 - pixel4[2] as i32).abs();

            let gy = (pixel1[0] as i32 - pixel2[0] as i32).abs() +
                     (pixel1[1] as i32 - pixel2[1] as i32).abs() +
                     (pixel1[2] as i32 - pixel2[2] as i32).abs();

            let g = ((gx.pow(2) + gy.pow(2)) as f64).sqrt();
            if g > threshold{
                new_pixel[0] = 255;
                new_pixel[1] = 255;
                new_pixel[2] = 255;
                new_image.put_pixel(x, y, image::Rgb(new_pixel));
            }
            else{
                new_pixel[0] = 0;
                new_pixel[1] = 0;
                new_pixel[2] = 0;
                new_image.put_pixel(x, y, image::Rgb(new_pixel));
            }
        }
    }
    *image = new_image;
    Ok(())
}
```



2.**Harris角点检测**：实现了Harris角点检测算法，用于识别图像中的角点特征。该算法首先对图像进行滤波处理，然后计算梯度和自相关矩阵，最后通过分析特征值来确定角点位置。

```rust
pub fn harris_detector(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>)  -> CvResult<()> {
    image_filter(image);
    let (width, height) = image.dimensions();
    let mut corner_image = ImageBuffer::new(width, height);
    // 定义卷积核
    let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    
    // 计算梯度的x和y方向分量
    let mut gx = vec![vec![0.0; height as usize]; width as usize];
    let mut gy = vec![vec![0.0; height as usize]; width as usize];
    
    // ...计算过程...
    
    // 检测角点
    for x in 1..width - 1 {
        for y in 1..height - 1 {
            if lambda1[x as usize][y as usize] > 200.0 && lambda2[x as usize][y as usize] > 200.0 {
                corner_image.put_pixel(x, y, image::Rgb(image.get_pixel(x, y).0));
            } else {
                corner_image.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
    }
    *image = corner_image;
    Ok(())
}
```



3.**图像可视化**：库还提供了将图像转换为ASCII艺术的功能，通过计算像素亮度并映射到不同字符，实现图像的文本表示，便于在终端环境中可视化图像内容。

```rust
pub fn print_image(image_buffer: &mut ImageBuffer<Rgb<u8>,Vec<u8>> ) -> String {
    let new_image = image::imageops::resize(image_buffer, 100, 50, image::imageops::FilterType::Nearest);
    *image_buffer = new_image;
    let char_list = [" ", ".", ":", "-", "=", "+", "*", "#"];
    let width = image_buffer.width();
    let height = image_buffer.height();
    let mut s = String::new();
    for y in 0..height {
        for x in 0..width {
            let pixel = image_buffer.get_pixel(x, y);
            let brightness = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;
            let index = (brightness / 255.0 * char_list.len() as f32).min(7.0) as usize;
            s.push_str(char_list[index]);
        }
        s.push('\n');
    }
    return s;
}
```

#### 3.3 图像变换功能

CV 库提供了多种图像变换功能，包括调整大小和马赛克效果：

1. **图像缩放**：提供了两种图像缩放函数，支持按比例调整图像大小。`resize_image_ratio`函数包含参数验证和错误处理，确保缩放操作的安全性和有效性。

   ```rust
   pub fn resize_image_ratio(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, ratio: f32) -> CvResult<()> {
       if ratio <= 0.0 {
           return Err(CvError::InvalidParameter(format!("缩放比例必须为正数，当前值: {}", ratio)));
       }    
       if ratio > 10.0 {
           return Err(CvError::InvalidParameter(format!("缩放比例过大，可能导致内存问题，当前值: {}", ratio)));
       }  
       let width = (image.width() as f32 * ratio) as u32;
       let height = (image.height() as f32 * ratio) as u32;
       if width == 0 || height == 0 {
           return Err(CvError::ProcessingError("缩放后的尺寸不能为零".to_string()));
       }
       let new_image = image::imageops::resize(image, width, height, image::imageops::FilterType::Nearest);
       *image = new_image;
       Ok(())
   }
   ```

   

2. **马赛克效果**：实现了马赛克效果处理，通过将图像分割成小块并使每个块内的像素值相同来创建像素化效果。用户可以通过调整因子参数来控制马赛克块的大小。

   ```rust
   pub fn mosaic(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,factor: u32) {
       // Implement mosaic effect
       let width = image_buffer.width();
       let height = image_buffer.height();
       for y in (0..height-factor).step_by(factor as usize) {
           for x in (0..width-factor).step_by(factor as usize) {           
               for y_offset in 0..factor {
                   for x_offset in 0..factor {
                       let pixel = image_buffer.get_pixel(x, y);
                       image_buffer.put_pixel(x + x_offset, y + y_offset, *pixel);
                   }
               }
           }
       }
   }
   ```

   

3. **图像滤波**：实现了均值滤波算法，通过计算像素邻域的平均值来平滑图像。该函数使用7×7的滑动窗口，对每个像素的RGB通道分别计算平均值，有效减少图像噪声。

   ```rust
   pub fn image_filter(image_buffer: &mut ImageBuffer<Rgb<u8>,Vec<u8>> ) -> CvResult<()> {
       let (width, height) = image_buffer.dimensions();   
       if width == 0 || height == 0 {
           return Err(CvError::InvalidParameter("图像尺寸不能为零".to_string()));
       } 
       let mut filtered_image = ImageBuffer::new(width, height);
       for x in 0..width {
           for y in 0..height {
               let mut sum_r = 0;
               let mut sum_g = 0;
               let mut sum_b = 0;
               let mut count = 0;        
               for dx in -3..=3 {
                   for dy in -3..=3 {
                       let nx = x as isize + dx;
                       let ny = y as isize + dy;                  
                       if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                           let pixel = image_buffer.get_pixel(nx as u32, ny as u32);
                           sum_r += pixel[0] as u32;
                           sum_g += pixel[1] as u32;
                           sum_b += pixel[2] as u32;
                           count += 1;
                       }
                   }
               } 
               if count == 0 {
                   return Err(CvError::ProcessingError("滤镜处理时计数为零".to_string()));
               }         
               let avg_r = (sum_r / count) as u8;
               let avg_g = (sum_g / count) as u8;
               let avg_b = (sum_b / count) as u8;       
               filtered_image.put_pixel(x, y, Rgb([avg_r, avg_g, avg_b]));
           }
       }
       *image_buffer = filtered_image;
       Ok(())
   }
   ```

   

4. **批处理支持**：从主程序中可以看出，库支持对多个图像进行批处理，通过多线程并行处理提高效率，适用于大量图像的批量转换和处理场景。

```rust
let handle = thread::spawn(move || {
    println!("Processing file: {}", path.display());
    let img = image::open(path).expect("Failed to open image");
    
    let mut img_buffer = img.to_rgb8();
    if resize_ratio != 1.0 {resize_image_ratio(&mut img_buffer, resize_ratio);}
    if noise {
        match add_noise(&mut img_buffer, 100)
        {
            Ok(_) => println!("Noise added successfully!"),
            Err(e) => println!("Error adding noise: {}", e),
        }
    }
    // 其他处理...
    img_buffer.save(output_img_path).expect("Failed to save image");
});
handles.push(handle);
```

所有这些功能都通过 Rust 的类型系统和错误处理机制确保了内存安全和运行时稳定性，同时保持了高效的性能表现。库的设计遵循模块化原则，使各功能可以单独使用或组合使用，满足不同的图像处理需求。

## 4. 界面设计与功能展示

​	软件使用 egui 框架实现了图形用户界面，提供了直观的交互体验：

![image-20250618193034994](C:\Users\86135\AppData\Roaming\Typora\typora-user-images\image-20250618193034994.png)

​	

```rust
impl MyEguiApp {

    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut fonts = egui::FontDefinitions::default();
        // 加载自定义字体
        fonts.font_data.insert(
            "my_font".to_owned(),
            egui::FontData::from_static(include_bytes!("fzdhtk gbk1 0.ttf")),
        );

        // 设置字体家族
        fonts.families.get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "my_font".to_owned());
        fonts.families.get_mut(&egui::FontFamily::Monospace)
            .unwrap()
            .insert(0, "my_font".to_owned());

        // 应用字体设置
        cc.egui_ctx.set_fonts(fonts);
    
        let mut style = (*cc.egui_ctx.style()).clone();
    
        // 设置不同文本样式的字体大小和字体家族
        style.text_styles = [
            (TextStyle::Heading, FontId::new(32.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(18.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(14.0, FontFamily::Monospace)),
            (TextStyle::Button, FontId::new(16.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
        ].into();
    
        // 设置间距
        style.spacing.item_spacing = egui::vec2(10.0, 10.0); // 控件之间的间距
        style.spacing.button_padding = egui::vec2(10.0, 5.0); // 按钮内边距
    
        // 应用样式
        cc.egui_ctx.set_style(style);

        Self { texture : None,
            img_path: "./input/flower.jpg".to_string() ,
            input_img_path : "./input/flower.jpg" .to_string() , 
            output_img_path : "./output/flower.jpg".to_string()  ,
            corner:false, edge:false, r:1.0, g:1.0, b:1.0, filter : false,
            mosaic:false, noise:false, sharp:0, resize_ratio:1.0, flag: false, err:-1,
            show_child_window: false,}
        
    }
    /// Loads and processes an image for preview
    /// 
    /// # Arguments
    /// 
    /// * `ctx` - The egui context
    /// 
    /// # Returns
    /// 
    /// Result indicating success or failure
    fn load_image(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let image_data = std::fs::read(&self.img_path)
            .map_err(|e| format!("failed to read file: {}", e))?;
        let mut image = image::load_from_memory(&image_data)
            .map_err(|e| format!("failed to decode image: {}", e))?
            .to_rgba8();
        resize_image(&mut image, 0.3);
        let size= [image.width() as usize, image.height() as usize];    
        let pixels = image.as_flat_samples();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            size,
            pixels.as_slice(),
        );
        // 创建纹理
        self.texture = Some(ctx.load_texture(
            "my-image",
            color_image,
            egui::TextureOptions::default()
        ));
        Ok(())
    }
}
```

​	软件支持包括角点检测、加噪、马赛克、锐化增强等多种功能，且可以直接输入图片所在位置进行处理；同时支持对文件夹进行批量处理，批量处理全部并行进行极大地提高了处理效率，点击下方按钮可以在软件中新弹出一个窗口查看图片处理的效果，方便快捷。

​	以下是对于一张图进行不同处理之后的效果展示：

<img src="C:\Users\86135\Desktop\rust\cv\src\output\flower.jpg" alt="flower" style="zoom:10%;" /><img src="C:\Users\86135\Desktop\rust\cv\src\output\flower1.jpg" alt="flower1" style="zoom:10%;" /><img src="C:\Users\86135\Desktop\rust\cv\src\output\flower2.jpg" alt="flower2" style="zoom:10%;" /><img src="C:\Users\86135\Desktop\rust\cv\src\output\flower3.jpg" alt="flower3" style="zoom:10%;" />

## 5. 总结与展望

### 5.1 项目总结

Rust-based CV 项目充分展示了 Rust 语言的核心优势：

1. **内存安全**：通过所有权系统和借用检查，在编译时防止内存错误和数据竞争
2. **错误处理**：使用类型系统进行全面的错误处理，避免运行时崩溃
3. **并发安全**：利用类型系统和所有权规则确保线程安全，高效利用多核处理器
4. **强大的类型系统**：通过枚举和模式匹配提供类型安全和代码清晰度
5. **零成本抽象**：高级抽象不带来运行时开销，保持高性能

​	这些特性使得本项目功能丰富，而且安全可靠，充分体现了 Rust 作为系统编程语言的优势，特别是在需要高性能和安全性的图像处理领域。
