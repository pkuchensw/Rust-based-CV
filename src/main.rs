use egui::accesskit::Node;
use image::load_from_memory_with_format;
use image::Rgb;
use rand::Rng;
use image::ImageBuffer;
use std::thread;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Condvar};
use eframe::egui;
use eframe::egui::IconData;
use egui::{ TextStyle, FontId, Style};
use egui::FontFamily;
use cv::{prints,add_noise,resize_image_ratio,mosaic,sharpen,edge_detection,harris_detector,image_filter};
use cv::{enhance_blue_channel,enhance_green_channel,enhance_red_channel,resize_image};

fn main() {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport.inner_size = Some(egui::vec2(800.0, 800.0));

    let icon_data = include_bytes!("./icon.png");
    let img = image::load_from_memory_with_format(icon_data, image::ImageFormat::Png).unwrap();
    let rgba_data = img.into_rgba8();
    let (w,h)=(rgba_data.width(),rgba_data.height());
    let raw_data: Vec<u8> = rgba_data.into_raw();
    native_options.viewport.icon=Some(Arc::<IconData>::new(IconData { rgba:  raw_data, width: w, height: h }));
    //native_options.viewport.fullscreen = Some(true);
    let _ = eframe::run_native("Image Processing App", 
        native_options, 
        Box::new(|cc| Box::new(MyEguiApp::new(cc)))
    );
    
    
        

}
pub struct MyEguiApp {
    /// Texture handle for displaying images
    texture: Option<egui::TextureHandle>,
    
    /// Current image path for preview
    img_path: String,
    
    /// Input path for batch processing
    input_img_path: String,
    output_img_path : String,
    corner: bool, edge:bool, r : f32 , g : f32 , b : f32, filter : bool,  mosaic: bool,
    noise : bool, sharp : i32, resize_ratio: f32 , flag: bool, err : i32
    , show_child_window: bool,
    
}

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

impl eframe::App for MyEguiApp {
    /// Updates the application state and renders the UI
    /// 
    /// # Arguments
    /// 
    /// * `ctx` - The egui context
    /// * `frame` - The application frame
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut style: Style = (*ctx.style()).clone();
        style.text_styles.insert(
            TextStyle::Body, 
            FontId::new(20.0, egui::FontFamily::Proportional), // 设置字体大小为 20.0
        );
        style.text_styles.insert(
            TextStyle::Heading, 
            FontId::new(40.0, egui::FontFamily::Proportional), 
        );
        style.text_styles.insert(
            TextStyle::Button, 
            FontId::new(20.0, egui::FontFamily::Proportional), 
        );
        ctx.set_style(style);
        egui::SidePanel::left("LeftBar").show(ctx,|ui|{
                ui.label("red channel");
                ui.add(egui::Slider::new(&mut self.r, 0.0..=2.0));
                ui.label("green channel");
                ui.add(egui::Slider::new(&mut self.g, 0.0..=2.0));
                ui.label("blue channel");
                ui.add(egui::Slider::new(&mut self.b, 0.0..=2.0));
                ui.add(egui::Checkbox::new(&mut self.corner,"corner detection"));
                ui.add(egui::Checkbox::new(&mut self.edge,"edge detection"));
                ui.add(egui::Checkbox::new(&mut self.mosaic,"mosaic"));
                ui.add(egui::Checkbox::new(&mut self.noise,"noise"));
                ui.add(egui::Checkbox::new(&mut self.filter,"filter"));
                
                if ui.button("reset").clicked(){
                    self.input_img_path = "./".to_string();
                    self.output_img_path = "./".to_string();
                    self.corner = false; self.edge = false;
                    self.r = 1.0; self.g = 1.0; self.b = 1.0;
                    self.filter = false;  self.mosaic = false;   self.noise = false;
                    self.sharp = 0; self.resize_ratio = 1.0;
                }
                
                
                
        });
        egui::CentralPanel::default()
        .show(ctx, |ui| {
            ui.image(egui::include_image!("./t(1).png")); 
                        
            ui.label("Please select the options at the left sidebar. \nYou can adjust the parameters for image processing. \nClick the button to generate or show the image.");
            ui.horizontal(|ui|{
                    if ui.button("Dark").clicked(){
                        ctx.set_visuals(egui::Visuals::dark());
                    }
                    if ui.button("Light").clicked(){
                        ctx.set_visuals(egui::Visuals::light());

                    }
                });
            ui.add_space(10.0);
            ui.add_space(10.0);
            ui.horizontal(|ui|{
                ui.label("input image path");
                ui.text_edit_singleline(&mut self.input_img_path);
            });
            ui.horizontal(|ui|{
                ui.label("output image path");
                ui.text_edit_singleline(&mut self.output_img_path);
            });
            ui.horizontal(|ui|{
                ui.label("Image path");
                ui.text_edit_singleline(&mut self.img_path);
            });
            ui.horizontal(|ui|{
        
                if ui.button("generate").clicked(){
                    ui.add_space(10.0);
                    self.err = process_image(self);
                    
                }
                if ui.button("show_picture").clicked(){
                    if self.flag {self.flag = false;}
                    else {self.flag = true;}
                    if let Err(err) = self.load_image(ctx) {
                        ui.label(format!("Error loading image."));
                    }
                    self.show_child_window = true;
                }
            });
            if self.err == 1{
                ui.label("Error: Input path is not a directory or file");
            }
            else if self.err == 2{
                ui.label("Error: Output path is not a directory or file");
            }
            else if self.err == -1{
                ui.label("Click generate button to process image");
            }
            else{
                ui.label("Image processed successfully! \nPlease check the output folder.");
            }
            if self.flag{       
              
                let mut  tmp = self.show_child_window;
                egui::Window::new("Image Window")
                .open(&mut tmp)
                .show(ctx, |ui| {
                    if ui.button("close").clicked() {
                        self.show_child_window = false;
                        self.flag = false;
                    }
                    if let Some(texture) = &self.texture {
                        ui.image(texture);      
                    }
                });
            }
            ui.image(egui::include_image!("./ep.png"));


            
            ui
            .with_layout(egui::Layout::right_to_left(egui::Align::BOTTOM), |ui| {
                
                if ui.button("Quit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                ui.label("Created by: 2300017751");
            });
            
        });
    }
    
}

/// Processes images based on the current application settings
/// 
/// # Arguments
/// 
/// * `f` - The application state containing processing parameters
/// 
/// # Returns
/// 
/// * `i32` - Status code (0 for success, non-zero for errors)
use std::time::Instant;

fn process_image(f: &MyEguiApp) -> i32 {
    let start = Instant::now();
    let mosiac =f.mosaic.clone();
    let noise = f.noise.clone();
    let red = f.r.clone();
    let green = f.g.clone();
    let blue = f.b.clone();
    let sharp = f.sharp.clone();
    let resize_ratio = f.resize_ratio.clone();
    let corner = f.corner.clone();
    let edge = f.edge.clone();
    let dir = f.input_img_path.clone();
    let out_dir = f.output_img_path.clone();
    let filter = f.filter.clone();

    let dir = Path::new(&dir);
    let out_dir = Path::new(&out_dir);
    
    if dir.is_dir() {
        if !out_dir.is_dir() {
            return 2;  
        } 
        let semaphore = Arc::new((Mutex::new(true), Condvar::new()));

        let mut handles = vec![];
        for entry in fs::read_dir(dir).expect("read_dir call failed") {
            let entry = entry.expect(" couldn't read entry");
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jpg") {
                let mut output_img_path = PathBuf::from(f.output_img_path.clone());
                if let Some(file_name) = path.file_stem() {
                    output_img_path.push(file_name);
                }
                output_img_path.set_extension("jpg"); // 更改文件扩展名

                println!("output_img_path: {}", output_img_path.display());

                let _semaphore = Arc::clone(&semaphore);
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
                    if red != 1.0 {enhance_red_channel(&mut img_buffer, red);}
                    if green != 1.0 {enhance_green_channel(&mut img_buffer, green);}
                    if blue != 1.0 {enhance_blue_channel(&mut img_buffer, blue);}
                    if mosiac {mosaic(&mut img_buffer , 30);}
                    if filter {image_filter(&mut img_buffer);}
                    if sharp == 1 {sharpen(&mut img_buffer);}
                    if edge {edge_detection(&mut img_buffer);}
                    if corner {harris_detector(&mut img_buffer);}
                    print!("Saving image... thread {:?}\n" ,thread::current().id());
                    img_buffer.save(output_img_path).expect("Failed to save image");

                });
                handles.push(handle);
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }

    } else if dir.is_file() &&
    ( dir.extension().and_then(|s| s.to_str()) == Some("jpg") 
    || dir.extension().and_then(|s| s.to_str()) == Some("png") ) {
        
        let mut output_img_path = PathBuf::from(f.output_img_path.clone());
        if  out_dir.extension().and_then(|s| s.to_str()) == Some("jpg") 
        || out_dir.extension().and_then(|s| s.to_str()) == Some("png") {
            output_img_path = out_dir.to_path_buf();
        }
        else if out_dir.is_dir() {
            if let Some(file_name) = dir.file_stem() {
                output_img_path.push(file_name);
            }
            output_img_path.set_extension("jpg"); 
                
        }
        else{
            return 2;
        }
        
        let img = image::open(dir).expect("Failed to open image");
        let mut img_buffer = img.to_rgb8();
        if resize_ratio != 1.0 {
            match resize_image_ratio(&mut img_buffer, resize_ratio)
            {
                Ok(_) => println!("Image resized successfully!"),
                Err(e) => println!("Error resizing image: {}", e),
            }
        }
        if noise {
            match add_noise(&mut img_buffer, 100)
            {
                Ok(_) => println!("Noise added successfully!"),
                Err(e) => println!("Error adding noise: {}", e),
            }
        }
        if red != 1.0 {
            match enhance_red_channel(&mut img_buffer, red){
                Ok(_) => println!("Red channel enhanced successfully!"),
                Err(e) => println!("Error enhancing red channel: {}", e),
            }
        }
        if green != 1.0 {
            match enhance_green_channel(&mut img_buffer, green){
                Ok(_) => println!("Green channel enhanced successfully!"),
                Err(e) => println!("Error enhancing green channel: {}", e),
            }
        }
        if blue != 1.0 {
            match enhance_blue_channel(&mut img_buffer, blue){
                Ok(_) => println!("Blue channel enhanced successfully!"),
                Err(e) => println!("Error enhancing blue channel: {}", e),
            }
        }
        if mosiac {mosaic(&mut img_buffer , 30);}
        if filter {image_filter(&mut img_buffer);}
        if sharp == 1 {sharpen(&mut img_buffer);}
        if edge {edge_detection(&mut img_buffer);}
        if corner {harris_detector(&mut img_buffer);}
        img_buffer.save(output_img_path).expect("Failed to save image");
    }
    else{
        println!("Not a directory or file");
        return 1;
    }
    let duration = start.elapsed();
    println!("Time elapsed in processing images: {:?}", duration);
    return 0;
}

