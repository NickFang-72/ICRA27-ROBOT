# X-ICM Geometry/Affordance Ablation Results

Scores are final success percentages. Paper rows use the mean values from the bundled X-ICM paper result summaries. Blank cells mean that ablation has not produced a strict `Finished ... Final Score` for that task yet. Bold marks the best available score in each task column.

## Completion

- original_xicm: 23/23 task scores
- geometry: 23/23 task scores
- affordance: 23/23 task scores
- geometry_affordance: 23/23 task scores
- geometry_affordance_v2_k6: 23/23 task scores
- geometry_affordance_v2_k8: 23/23 task scores
- geometry_affordance_v2_k10: 23/23 task scores

## Task Scores

| Method | put_toilet_roll_on_stand | put_knife_on_chopping_board | close_fridge | close_microwave | close_laptop_lid | phone_on_base | toilet_seat_down | lamp_off | lamp_on | put_books_on_bookshelf | put_umbrella_in_umbrella_stand | open_grill | put_rubbish_in_bin | take_usb_out_of_computer | take_lid_off_saucepan | take_plate_off_colored_dish_rack | basketball_in_hoop | scoop_with_spatula | straighten_rope | turn_oven_on | beat_the_buzz | water_plants | unplug_charger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| X-ICM 7B (paper) | 1.33 | 26.67 | 22.67 | 45.33 | 33.33 | 57.33 | 48 | 58.67 | 50.67 | 1.33 | 0 | 8 | **18.67** | **98.67** | **20** | **6.67** | 9.33 | 0 | 6.67 | 16 | 2.67 | 5.33 | 4 |
| X-ICM 72B (paper) | **6.67** | **69.33** | 12.67 | 58.67 | 34 | **68** | 51.33 | **86.67** | 74.67 | 2 | 1.33 | 5.33 | **18.67** | **98.67** | 13.33 | 4.67 | **36** | **0.67** | **16** | 20.67 | 7.33 | 2.67 | 2.67 |
| X-ICM 7B rerun | 0 | 20 | 20 | 48 | **40** | 56 | 60 | 60 | 40 | 0 | 0 | 4 | 16 | 96 | 16 | 0 | 4 | 0 | 8 | 16 | 0 | 0 | 4 |
| + geometry | 0 | 8 | 4 | 36 | 32 | 20 | 64 | 84 | 52 | 0 | 4 | 12 | 0 | 88 | 0 | 0 | 0 | 0 | 8 | **40** | 0 | **12** | 0 |
| + affordance | 0 | 8 | **24** | 56 | 8 | 12 | **76** | 48 | 56 | **4** | 0 | **24** | 0 | 88 | 8 | 0 | 0 | 0 | **16** | 24 | 4 | **12** | **8** |
| + geometry + affordance | 0 | 16 | 12 | 32 | 28 | 8 | 60 | 76 | **76** | 0 | 4 | **24** | 0 | 96 | 8 | 0 | 0 | 0 | 8 | 28 | 4 | 0 | 0 |
| + geometry + affordance v2 k=6 | 0 | 8 | 4 | 64 | 36 | 4 | 68 | 84 | **76** | 0 | 4 | 20 | 8 | 84 | 0 | 0 | 0 | 0 | 4 | 28 | 0 | 8 | 0 |
| + geometry + affordance v2 k=8 | 0 | 16 | 4 | **72** | 20 | 0 | 72 | 80 | 72 | 0 | **8** | **24** | 4 | 92 | 4 | 0 | 0 | 0 | 0 | 16 | **12** | 0 | 4 |
| + geometry + affordance v2 k=10 | 0 | 20 | 8 | 40 | 28 | 8 | 72 | 72 | 72 | 0 | 0 | 20 | 0 | 92 | 8 | 4 | 0 | 0 | 8 | 20 | 4 | 0 | 0 |

## Summary

| Method | Level 1 Avg | Level 2 Avg | Average |
| --- | ---: | ---: | ---: |
| X-ICM 7B (paper) | 28.62 | 16.93 | 23.54 |
| X-ICM 72B (paper) | **37.64** | **20.27** | **30.09** |
| X-ICM 7B rerun | 28 | 14.4 | 22.09 |
| + geometry | 24.31 | 14.8 | 20.17 |
| + affordance | 24.31 | 16 | 20.7 |
| + geometry + affordance | 25.85 | 14.4 | 20.87 |
| + geometry + affordance v2 k=6 | 28.92 | 12.4 | 21.74 |
| + geometry + affordance v2 k=8 | 28.62 | 12.8 | 21.74 |
| + geometry + affordance v2 k=10 | 26.15 | 13.6 | 20.7 |
