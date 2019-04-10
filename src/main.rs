mod seam_carving;

use clap::{App, Arg};
use gifski::{progress::ProgressBar, Settings};
use image::{DynamicImage, FilterType, GenericImageView};
use imgref::Img;
use rayon::prelude::*;
use rgb::RGBA8;
use std::{
    fs::OpenOptions,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

fn main() {
    let clap = App::new("Animated seam carving")
        .author("Authors: DasEtwas, Friz64")
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .value_name("FILE")
                .help("Input file")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("FILE")
                .default_value("output.gif")
                .help("Output file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("length")
                .short("l")
                .long("length")
                .value_name("SECONDS")
                .default_value("1")
                .help("GIF length in seconds")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("scale")
                .short("s")
                .long("scale")
                .value_name("PERCENT")
                .default_value("10")
                .help("How much to downscale the image")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("fps")
                .short("f")
                .long("fps")
                .value_name("FPS")
                .default_value("20")
                .help("GIF FPS")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("quality")
                .short("q")
                .long("quality")
                .value_name("QUALITY")
                .default_value("30")
                .help("GIF Quality [1 - 100]")
                .takes_value(true),
        )
        .get_matches();

    let input = clap.value_of("input").unwrap();
    let output = clap.value_of("output").unwrap();
    let length: f32 = match clap.value_of("length").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing fps: {}", err);
            return;
        }
    };
    let scale: f32 = match clap.value_of("scale").unwrap().parse::<f32>() {
        Ok(val) => val / 100.0,
        Err(err) => {
            println!("Error while parsing scale: {}", err);
            return;
        }
    };
    let fps: f32 = match clap.value_of("fps").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing fps: {}", err);
            return;
        }
    };
    let quality: u8 = match clap.value_of("quality").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing quality: {}", err);
            return;
        }
    };

    let frames = fps * length;
    let delay = ((1.0 / fps) * 10.0).round() as u16;

    let image = match image::open(input) {
        Ok(val) => val,
        Err(err) => {
            println!("Failed to open input: {}", err);
            return;
        }
    };

    let dimensions = image.dimensions();
    let width = dimensions.0 as f32;
    let height = dimensions.1 as f32;

    let (collector, writer) = gifski::new(Settings {
        width: Some(dimensions.0),
        height: Some(dimensions.1),
        quality,
        once: false,
        fast: false,
    })
    .expect("Failed to create encoder");

    let collector = Arc::new(Mutex::new(collector));

    println!("Calculating...");

    let start = Instant::now();
    thread::spawn(move || {
        (0..frames.round() as usize).into_par_iter().for_each({
            let collector = collector.clone();

            move |i| {
                let new_width = lerp(width, width * scale, i as f32 / frames as f32) as u32;
                let new_height = lerp(height, height * scale, i as f32 / frames as f32) as u32;

                let frame_image = seam_carving::resize(&image, new_width, new_height).resize_exact(
                    width as u32,
                    height as u32,
                    FilterType::Nearest,
                );
                let frame = image_to_frame(&frame_image);

                collector
                    .lock()
                    .unwrap()
                    .add_frame_rgba(i, frame, delay)
                    .expect("Failed to add frame");
            }
        });
    });


    let output_result = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output);
    let output = match output_result {
        Ok(val) => val,
        Err(err) => {
            println!("Failed to open output: {}", err);
            return;
        }
    };

    let mut progress_bar = ProgressBar::new(frames.round() as u64);

    writer
        .write(output, &mut progress_bar)
        .expect("Failed to write output");

    println!("Done in {:?}!", start.elapsed());
}

fn lerp(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

fn image_to_frame(image: &DynamicImage) -> Img<Vec<RGBA8>> {
    let (width, height) = image.dimensions();

    let container: Vec<RGBA8> = image
        .to_rgba()
        .into_raw()
        .chunks(4)
        .map(|chunk| RGBA8::new(chunk[0], chunk[1], chunk[2], chunk[3]))
        .collect();

    Img::new(container, width as usize, height as usize)
}