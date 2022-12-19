## Generate Experimental Stimuli and Process Subjective Eye Movement Data

### Generate Experimental Stimuli

- Generate AR images (size: 1440*900)
```
preprocess_generate_ar_img.m
```
- Generate BG images (equirectangular, w/o fixed size)
```
preprocess_generate_bg_img.m
```
- Generate sequence for unity experiment
```
preprocess_generate_seq.m
```
- Wrap equirectangular BG images to cubic and crop the front image (cubic size: 2560*2560)
```
preprocess_generate_bg_cubic.m
```
- Pad ar images with zeros to the same size with one face of bg cubics (cubic size: 2560*2560)
```
preprocess_generate_ar_pad.m
```

### Process Subjective Eye Movement Data

- Process captured mixed image data (crop and generate crop part)
```
process_mixed_img.m
```

- Copy and organize all raw eye tracking data into one folder
```
process_data.m
```

- **Generate fixations from raw gaze data**
```
process_generate_eyefixation.m
```

- Change name of all files to new order
```
process_rewrite_names.m
```

- **Generate fixation_points_map, fixation_map, heat_map**
```
process_generate_eyemaps.m
```

- Resize to small size for model prediction
```
process_resize.m
```

- Delete suffix "_fixpts" or "_fixmap" of small size images
```
process_resize_deleteSuffix.m
```