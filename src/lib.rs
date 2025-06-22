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
//! 
//! ## Quick Start
//! 
//! ```rust
//! use cv::{resize_image_ratio, add_noise, enhance_red_channel};
//! use image::ImageBuffer;
//! 
//! // Load and process an image
//! let mut img = image::open("input.jpg").unwrap().to_rgb8();
//! 
//! // Resize to 50%
//! resize_image_ratio(&mut img, 0.5);
//! 
//! // Add some noise
//! add_noise(&mut img, 100);
//! 
//! // Enhance red channel
//! enhance_red_channel(&mut img, 1.5);
//! ```

use image::{ImageBuffer, Rgb};
use rand::Rng;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::Path;

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

impl fmt::Display for CvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CvError::IoError(err) => write!(f, "IO错误: {}", err),
            CvError::ImageError(err) => write!(f, "图像处理错误: {}", err),
            CvError::InvalidParameter(msg) => write!(f, "参数无效: {}", msg),
            CvError::PathError(msg) => write!(f, "路径错误: {}", msg),
            CvError::ProcessingError(msg) => write!(f, "处理错误: {}", msg),
        }
    }
}

impl Error for CvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CvError::IoError(err) => Some(err),
            CvError::ImageError(err) => Some(err),
            _ => None,
        }
    }
}

// 从标准IO错误转换
impl From<io::Error> for CvError {
    fn from(err: io::Error) -> Self {
        CvError::IoError(err)
    }
}

// 从图像处理错误转换
impl From<image::ImageError> for CvError {
    fn from(err: image::ImageError) -> Self {
        CvError::ImageError(err)
    }
}

/// 结果类型别名，用于简化函数签名
pub type CvResult<T> = Result<T, CvError>;

/// 打印测试消息用于调试
pub fn prints() {
    println!("Hello, world!");
}

/// Converts and prints an image as ASCII art
/// 
/// # Arguments
/// 
/// * `image_buffer` - The input image buffer to convert
/// 
/// # Returns
/// 
/// * `String` - ASCII representation of the image
/// 
/// # Examples
/// 
/// ```
/// use image::{ImageBuffer, Rgb};
/// use cv::print_image;
/// 
/// let mut img = ImageBuffer::new(100, 100);
/// let ascii = print_image(&mut img);
/// println!("{}", ascii);
/// ```


/// Adds random noise to an image
/// 
/// # Arguments
/// 
/// * `image_buffer` - The image to add noise to
/// * `intensity` - Noise intensity (0-255)
/// 
/// # Examples
/// 
/// ```
/// use image::{ImageBuffer, Rgb};
/// use cv::add_noise;
/// 
/// # 返回值
/// 
/// * `Result<(), CvError>` - 成功时返回空元组，失败时返回错误
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
/// Resizes an image by a given ratio
/// 
/// # Arguments
/// 
/// * `image_buffer` - The image to resize
/// * `ratio` - Scaling ratio (0.1 to 10.0)
/// 
/// # Panics
/// Panics if ratio is 0.0

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

pub fn resize_image(image: &mut ImageBuffer<image::Rgba<u8>, Vec<u8>>, ratio: f32) {
    let width = (image.width() as f32 * ratio) as u32;
    let height = (image.height() as f32 * ratio) as u32;
    let new_image = image::imageops::resize(image, width, height, image::imageops::FilterType::Nearest);
    *image = new_image;
}
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
pub fn enhance_green_channel(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, factor: f32) -> CvResult<()> {
    if factor < 0.0 || factor > 2.0 {
        return Err(CvError::InvalidParameter(format!("增强因子必须在0.0-2.0之间，当前值: {}", factor)));
    }
    
    for (_x, _y, pixel) in image_buffer.enumerate_pixels_mut() {
        let new_g = (pixel[1] as f32 * factor).min(255.0) as u8;
        *pixel = Rgb([pixel[0], new_g, pixel[2]]);
    }
    
    Ok(())
}
pub fn enhance_blue_channel(image_buffer: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, factor: f32) -> CvResult<()> {
    if factor < 0.0 || factor > 2.0 {
        return Err(CvError::InvalidParameter(format!("增强因子必须在0.0-2.0之间，当前值: {}", factor)));
    }
    
    for (_x, _y, pixel) in image_buffer.enumerate_pixels_mut() {
        let new_b = (pixel[2] as f32 * factor).min(255.0) as u8;
        *pixel = Rgb([pixel[0], pixel[1], new_b]);
    }
    
    Ok(())
}
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
pub fn sharpen(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>) {
    // Implement edge detection
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
                //new_image.put_pixel(x, y, image::Rgb(image.get_pixel(x, y).0));
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

    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    let pixel0 = image.get_pixel(x + i - 1, y + j - 1).0[0] as f64;
                    let pixel1 = image.get_pixel(x + i - 1, y + j - 1).0[1] as f64;
                    let pixel2 = image.get_pixel(x + i - 1, y + j - 1).0[2] as f64;
                    let pixel = (pixel0 + pixel1 + pixel2) / 3.0;
                    sum_x += pixel * sobel_x[(i * 3 + j)as usize];
                    sum_y += pixel * sobel_y[(i * 3 + j)as usize];
                }
            }
            gx[x as usize][y as usize] = sum_x;
            gy[x as usize][y as usize] = sum_y;
        }
    }

    // 计算梯度的乘积和梯度的平方
    let mut gxx = vec![vec![0.0; height as usize]; width as usize];
    let mut gyy = vec![vec![0.0; height as usize]; width as usize];
    let mut gxy = vec![vec![0.0; height as usize]; width as usize];

    for x in 1..width - 1 {
        for y in 1..height - 1 {
            gxx[x as usize][y as usize] = gx[x as usize][y as usize].powi(2);
            gyy[x as usize][y as usize] = gy[x as usize][y as usize].powi(2);
            gxy[x as usize][y as usize] = gx[x as usize][y as usize] * gy[x as usize][y as usize];
        }
    }

    // 计算每个像素点的自相关矩阵M
    let mut m = vec![vec![[0.0; 4]; height as usize]; width as usize];

    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let mut sum_gxx = 0.0;
            let mut sum_gyy = 0.0;
            let mut sum_gxy = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    sum_gxx += gxx[(x + i - 1) as usize][(y + j - 1) as usize];
                    sum_gyy += gyy[(x + i - 1) as usize][(y + j - 1) as usize];
                    sum_gxy += gxy[(x + i - 1) as usize][(y + j - 1) as usize];
                }
            }
            m[x as usize][y as usize] = [sum_gxx / 9.0, sum_gxy / 9.0, sum_gxy / 9.0, sum_gyy / 9.0];
        }
    }

    // 计算特征值λ1和λ2
    let mut lambda1 = vec![vec![0.0; height as usize]; width as usize];
    let mut lambda2 = vec![vec![0.0; height as usize]; width as usize];

    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let m11 = m[x as usize][y as usize][0];
            let m12 = m[x as usize][y as usize][1];
            let m21 = m[x as usize][y as usize][2];
            let m22 = m[x as usize][y as usize][3];

            let trace = m11 + m22;
            let det = m11 * m22 - m12 * m21;

            lambda1[x as usize][y as usize] = (trace + (trace.powi(2) - 4.0 * det).sqrt()) / 2.0;
            lambda2[x as usize][y as usize] = (trace - (trace.powi(2) - 4.0 * det).sqrt()) / 2.0;
        }
    }

    // 计算角点响应函数R
    let k = 0.045;
    let t = 0.1;
    let mut r = vec![vec![0.0; height as usize]; width as usize];

    for x in 1..width - 1 {
        for y in 1..height - 1 {
            let l1 = lambda1[x as usize][y as usize];
            let l2 = lambda2[x as usize][y as usize];

            r[x as usize][y as usize] = l1 * l2 - k * (l1 + l2).powi(2) - t ;
        }
    }
    // 绘制角点
    
    for x in 1..width - 1 {
        for y in 1..height - 1 {
            
            //if r[x as usize][y as usize] > 0.0 {
            if lambda1[x as usize][y as usize] > 200.0 && lambda2[x as usize][y as usize] > 200.0 {
                //corner_image.put_pixel(x, y, Rgb([0, 0, 0]));
                corner_image.put_pixel(x, y, image::Rgb(image.get_pixel(x, y).0));
            } else {
                corner_image.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }

    }
    *image = corner_image;
    Ok(())

}
