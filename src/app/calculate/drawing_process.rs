use crate::app::SeedColor;
use crate::app::calculate;
use crate::app::preset::UnprocessedPreset;

use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, atomic::AtomicU32, mpsc};

#[cfg(not(target_arch = "wasm32"))]
use super::ProgressMsg;

use super::GenerationSettings;

#[derive(Clone, Copy)]
pub struct PixelData {
    pub stroke_id: u32,
    pub last_edited: u32,
}
impl PixelData {
    pub(crate) fn init_canvas(frame_count: u32) -> Vec<PixelData> {
        vec![
            PixelData {
                stroke_id: 0,
                last_edited: frame_count
            };
            DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE
        ]
    }
}

pub const DRAWING_CANVAS_SIZE: usize = 128;

use super::heuristic;

#[derive(Clone, Copy)]
pub struct DrawingParams {
    pub stroke_reward: i64,
    pub max_dist_base: u32,
    pub max_dist_decay: f32,
    pub max_dist_min: u32,
}

impl DrawingParams {
    pub fn max_dist(&self, age: u32) -> u32 {
        let decay_steps = (age / 30) as i32;
        let raw = (self.max_dist_base as f32) * self.max_dist_decay.powi(decay_steps);
        raw.round().max(self.max_dist_min as f32) as u32
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DrawingPixel {
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) h: i64, // base heuristic value (no stroke reward)
}

impl DrawingPixel {
    pub(crate) fn new(src_x: u16, src_y: u16, h: i64) -> Self {
        Self { src_x, src_y, h }
    }

    pub(crate) fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    pub(crate) fn calc_drawing_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        colors: &[SeedColor],
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            {
                let rgba =
                    colors[self.src_y as usize * DRAWING_CANVAS_SIZE + self.src_x as usize].rgba;
                (
                    (rgba[0] * 256.0) as u8,
                    (rgba[1] * 256.0) as u8,
                    (rgba[2] * 256.0) as u8,
                )
            },
            target_col,
            weight,
            proximity_importance,
        )
    }
}

pub struct DrawingState {
    pixels: Vec<DrawingPixel>,
    rng: frand::Rand,
    settings: GenerationSettings,
    target_pixels: Vec<(u8, u8, u8)>,
    weights: Vec<i64>,
}

impl DrawingState {
    pub fn new(
        source: UnprocessedPreset,
        settings: GenerationSettings,
        colors: &[SeedColor],
        _params: &DrawingParams,
    ) -> Result<Self, Box<dyn Error>> {
        let source_img = image::ImageBuffer::from_raw(
            source.width,
            source.height,
            source.source_img.clone(),
        )
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid source image"))?;
        let (source_pixels, target_pixels, weights) =
            calculate::util::get_images(source_img, &settings)?;

        let pixels = source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % settings.sidelen) as u16;
                let y = (i as u32 / settings.sidelen) as u16;
                let mut p = DrawingPixel::new(x, y, 0);
                let h = p.calc_drawing_heuristic(
                    (x, y),
                    target_pixels[i],
                    weights[i],
                    colors,
                    settings.proximity_importance,
                );
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>();

        Ok(Self {
            pixels,
            rng: frand::Rand::with_seed(12345),
            settings,
            target_pixels,
            weights,
        })
    }

    pub fn step(
        &mut self,
        colors: &[SeedColor],
        pixel_data: &[PixelData],
        frame_count: u32,
        max_swaps: usize,
        params: &DrawingParams,
    ) -> Option<Vec<usize>> {
        let mut swaps_made = 0;

        for _ in 0..max_swaps {
            let apos = self.rng.gen_range(0..self.pixels.len() as u64) as usize;
            let ax = apos as u16 % self.settings.sidelen as u16;
            let ay = apos as u16 / self.settings.sidelen as u16;

            let max_dist_a = params.max_dist(frame_count.saturating_sub(pixel_data[apos].last_edited));

            let bx =
                (ax as i16 + self.rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                    .clamp(0, self.settings.sidelen as i16 - 1) as u16;
            let by =
                (ay as i16 + self.rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                    .clamp(0, self.settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * self.settings.sidelen as usize + bx as usize;

            let max_dist_b = params.max_dist(frame_count.saturating_sub(pixel_data[bpos].last_edited));
            if (bx as i32 - ax as i32).abs() > max_dist_b as i32
                || (by as i32 - ay as i32).abs() > max_dist_b as i32
            {
                continue;
            }

            let t_a = self.target_pixels[apos];
            let t_b = self.target_pixels[bpos];

            let current_a = self.pixels[apos].h
                + stroke_reward_with_params(apos, apos, pixel_data, &self.pixels, frame_count, params);
            let current_b = self.pixels[bpos].h
                + stroke_reward_with_params(bpos, bpos, pixel_data, &self.pixels, frame_count, params);

            let a_on_b_base = self.pixels[apos].calc_drawing_heuristic(
                (bx, by),
                t_b,
                self.weights[bpos],
                colors,
                self.settings.proximity_importance,
            );
            let b_on_a_base = self.pixels[bpos].calc_drawing_heuristic(
                (ax, ay),
                t_a,
                self.weights[apos],
                colors,
                self.settings.proximity_importance,
            );
            let a_on_b_h = a_on_b_base
                + stroke_reward_with_params(bpos, apos, pixel_data, &self.pixels, frame_count, params);

            let b_on_a_h = b_on_a_base
                + stroke_reward_with_params(apos, bpos, pixel_data, &self.pixels, frame_count, params);

            let improvement_a = current_a - b_on_a_h;
            let improvement_b = current_b - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                self.pixels.swap(apos, bpos);
                self.pixels[apos].update_heuristic(b_on_a_base);
                self.pixels[bpos].update_heuristic(a_on_b_base);
                swaps_made += 1;
            }
        }

        if swaps_made > 0 {
            Some(
                self.pixels
                    .iter()
                    .map(|p| {
                        p.src_y as usize * self.settings.sidelen as usize + p.src_x as usize
                    })
                    .collect(),
            )
        } else {
            None
        }
    }
}

pub(crate) fn stroke_reward_with_params(
    newpos: usize,
    oldpos: usize,
    pixel_data: &[PixelData],
    pixels: &[DrawingPixel],
    frame_count: u32,
    params: &DrawingParams,
) -> i64 {
    let x = (newpos % DRAWING_CANVAS_SIZE) as u16;
    let y = (newpos / DRAWING_CANVAS_SIZE) as u16;
    // look at 8-connected neighbors
    // if any has the same stroke_id, return true
    let data = pixel_data
        [pixels[oldpos].src_x as usize + pixels[oldpos].src_y as usize * DRAWING_CANVAS_SIZE];
    let stroke_id = data.stroke_id;
    let _age = frame_count - data.last_edited;

    for (dx, dy) in [
        //(-1, -1),
        (0, -1),
        //(1, -1),
        (-1, 0),
        (1, 0),
        //(-1, 1),
        (0, 1),
        //(1, 1),
    ] {
        let nx = x as i16 + dx;
        let ny = y as i16 + dy;
        if nx < 0 || nx >= DRAWING_CANVAS_SIZE as i16 || ny < 0 || ny >= DRAWING_CANVAS_SIZE as i16
        {
            continue;
        }
        let npos = ny as usize * DRAWING_CANVAS_SIZE + nx as usize;
        if pixel_data
            [pixels[npos].src_x as usize + pixels[npos].src_y as usize * DRAWING_CANVAS_SIZE]
            .stroke_id
            == stroke_id
        {
            return params.stroke_reward;
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "wasm32"))]
pub fn drawing_process_genetic(
    source: UnprocessedPreset,
    settings: GenerationSettings,
    tx: mpsc::SyncSender<ProgressMsg>,
    colors: Arc<std::sync::RwLock<Vec<SeedColor>>>,
    pixel_data: Arc<std::sync::RwLock<Vec<PixelData>>>,
    frame_count: u32,
    my_id: u32,
    current_id: Arc<AtomicU32>,
    params: DrawingParams,
) -> Result<(), Box<dyn Error>> {
    let source_img =
        image::ImageBuffer::from_raw(source.width, source.height, source.source_img.clone())
            .unwrap();
    let (source_pixels, target_pixels, weights) =
        calculate::util::get_images(source_img, &settings)?;

    let mut pixels = {
        let read_colors: Vec<SeedColor> = colors.read().unwrap().clone();
        //let read_pixel_data: Vec<PixelData> = pixel_data.read().unwrap().clone();

        source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % settings.sidelen) as u16;
                let y = (i as u32 / settings.sidelen) as u16;
                let mut p = DrawingPixel::new(x, y, 0);
                let h = p.calc_drawing_heuristic(
                    (x, y),
                    target_pixels[i],
                    weights[i],
                    &read_colors,
                    settings.proximity_importance,
                    // &read_pixel_data,
                );
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>()
    };

    let mut rng = frand::Rand::with_seed(12345);
    let swaps_per_generation = SWAPS_PER_GENERATION_PER_PIXEL * pixels.len();

    loop {
        let colors: Vec<SeedColor> = {
            let r = colors.read().unwrap();
            r.clone()
        };
        let pixel_data = {
            let r = pixel_data.read().unwrap();
            r.clone()
        };
        let mut swaps_made = 0;

        for _ in 0..swaps_per_generation {
            let apos = rng.gen_range(0..pixels.len() as u64) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;

            //let stroke_id = pixel_data[apos].stroke_id as usize;
            let max_dist_a = params.max_dist(frame_count.saturating_sub(pixel_data[apos].last_edited));

            let bx = (ax as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;

            let max_dist_b = params.max_dist(frame_count.saturating_sub(pixel_data[bpos].last_edited));
            if (bx as i32 - ax as i32).abs() > max_dist_b as i32
                || (by as i32 - ay as i32).abs() > max_dist_b as i32
            {
                continue;
            }

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let current_a = pixels[apos].h
                + stroke_reward_with_params(apos, apos, &pixel_data, &pixels, frame_count, &params);
            let current_b = pixels[bpos].h
                + stroke_reward_with_params(bpos, bpos, &pixel_data, &pixels, frame_count, &params);

            let a_on_b_base = pixels[apos].calc_drawing_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                &colors,
                settings.proximity_importance,
            );

            let b_on_a_base = pixels[bpos].calc_drawing_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                &colors,
                settings.proximity_importance,
            );
            let a_on_b_h = a_on_b_base
                + stroke_reward_with_params(bpos, apos, &pixel_data, &pixels, frame_count, &params);
            let b_on_a_h = b_on_a_base
                + stroke_reward_with_params(apos, bpos, &pixel_data, &pixels, frame_count, &params);

            let improvement_a = current_a - b_on_a_h;
            let improvement_b = current_b - a_on_b_h;
            if improvement_a + improvement_b > 0 {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_base);
                pixels[bpos].update_heuristic(a_on_b_base);
                swaps_made += 1;
            }
        }

        //println!("swaps made: {}", swaps_made);

        // let img = make_new_img(&source_pixels, &assignments, target.width());
        // if swaps_made < 10 || cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        //     let dir_name = save_result(target, base_name, source, assignments, img)?;
        //     tx.send(ProgressMsg::Done(PathBuf::from(format!(
        //         "./presets/{}",
        //         dir_name
        //     ))))?;
        //     return Ok(());
        // }
        // tx.send(ProgressMsg::UpdatePreview(img))?;
        if swaps_made > 0 {
            let assignments = pixels
                .iter()
                .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
                .collect::<Vec<_>>();
            tx.send(ProgressMsg::UpdateAssignments(assignments))?;
        }
        if my_id != current_id.load(std::sync::atomic::Ordering::Relaxed) {
            tx.send(ProgressMsg::Cancelled).unwrap();
            return Ok(());
        }

        //max_dist = (max_dist as f32 * 0.99).max(4.0) as u32;
    }
}
