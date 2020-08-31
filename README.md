# seam carving
Library for seamcarving images

## Usage
Depends [image 0.23](https://crates.io/crates/image)

```
let img = image::open("test.png")?;
let resized = seam_carving::easy_resize(&img, 1920, 1080); // Can be anything you want
resized.save("test-r.png")?;
```

## To-Do
- Improve Performance \
 *We are currently 3x slower than ImageMagick's implementation*
