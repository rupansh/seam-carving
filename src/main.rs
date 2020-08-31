pub mod seam_carving;

fn main() {
    let input = "test.png";
    let output = "testd.png";

    let image = match image::open(input) {
        Ok(val) => val,
        Err(err) => {
            println!("Failed to open input: {}", err);
            return;
        }
    };

    let frame_image = seam_carving::easy_resize(&image, 320, 320);
    frame_image.save(output).unwrap();
}
