# Geometry/Affordance Human Review Index

Pilot outputs for human checking. Qwen2.5-VL is used only for geometry descriptions; RoboPoint is used only for affordance/contact/keypoint descriptions.

## turn_tap_episode161_71_143

- Task: turn left tap
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/turn_tap_episode161_71_143`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/turn_tap_episode161_71_143/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['tap', 'handle', 'knob', 'rotational_axis', 'cylindrical']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/turn_tap_episode161_71_143/affordance_robopoint.json` (done)
- Affordance labels: grasp=`knob_grasp`, contact=`rotate_part`, motion=`rotate`, region=`tap_handle_or_knob`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/turn_tap_episode161_71_143/combined_review.json`
- RoboPoint preferred contact points: `[[0.3, 0.59], [0.302, 0.548], [0.303, 0.498], [0.303, 0.456]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/turn_tap_episode161_71_143/view0_71.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/turn_tap_episode161_71_143/view1_71.png`

## close_jar_episode161_65_99

- Task: close the red jar
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/close_jar_episode161_65_99`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/close_jar_episode161_65_99/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['jar', 'round', 'cylindrical', 'hollow', 'lid', 'rim', 'top_opening']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/close_jar_episode161_65_99/affordance_robopoint.json` (done)
- Affordance labels: grasp=`rim_grasp`, contact=`rotate_part`, motion=`twist`, region=`lid_or_rim`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/close_jar_episode161_65_99/combined_review.json`
- RoboPoint preferred contact points: `[[0.498, 0.6], [0.534, 0.596], [0.467, 0.602], [0.5, 0.558]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/close_jar_episode161_65_99/view0_65.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/close_jar_episode161_65_99/view1_65.png`

## light_bulb_in_episode161_37_51

- Task: screw in the red light bulb
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/light_bulb_in_episode161_37_51`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/light_bulb_in_episode161_37_51/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['light_bulb', 'round', 'bulb', 'threaded_base', 'socket', 'rotational_axis', 'alignment_sensitive']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/light_bulb_in_episode161_37_51/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`insert_object`, motion=`insert_then_twist`, region=`bulb_body_or_base`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/light_bulb_in_episode161_37_51/combined_review.json`
- RoboPoint preferred contact points: `[[0.63, 0.6], [0.631, 0.65], [0.662, 0.61], [0.6, 0.61], [0.662, 0.66]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/light_bulb_in_episode161_37_51/view0_37.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/light_bulb_in_episode161_37_51/view1_37.png`

## slide_block_to_color_target_episode161_67_87

- Task: slide the block to green target
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/slide_block_to_color_target_episode161_67_87`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/slide_block_to_color_target_episode161_67_87/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['block', 'rectangular', 'flat_faces', 'solid', 'target_region']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/slide_block_to_color_target_episode161_67_87/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`push_surface`, motion=`slide`, region=`block_side_or_top`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/slide_block_to_color_target_episode161_67_87/combined_review.json`
- RoboPoint preferred contact points: `[[0.398, 0.69], [0.397, 0.648], [0.367, 0.671], [0.366, 0.629]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/slide_block_to_color_target_episode161_67_87/view0_67.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/slide_block_to_color_target_episode161_67_87/view1_67.png`

## sweep_to_dustpan_of_size_episode161_57_68

- Task: sweep dirt to the tall dustpan
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/sweep_to_dustpan_of_size_episode161_57_68`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/sweep_to_dustpan_of_size_episode161_57_68/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['dustpan', 'open_container', 'thin_edge', 'flat_floor', 'target_region']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/sweep_to_dustpan_of_size_episode161_57_68/affordance_robopoint.json` (done)
- Affordance labels: grasp=`handle_grasp`, contact=`push_surface`, motion=`push`, region=`sweeping_tool_or_object_side`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/sweep_to_dustpan_of_size_episode161_57_68/combined_review.json`
- RoboPoint preferred contact points: `[[0.386, 0.09], [0.389, 0.131], [0.388, 0.048], [0.388, 0.173]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/sweep_to_dustpan_of_size_episode161_57_68/view0_57.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/sweep_to_dustpan_of_size_episode161_57_68/view1_57.png`

## push_buttons_episode161_51_61

- Task: push the maroon button
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/push_buttons_episode161_51_61`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/push_buttons_episode161_51_61/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['button', 'round', 'small', 'raised_surface', 'flat_top']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/push_buttons_episode161_51_61/affordance_robopoint.json` (done)
- Affordance labels: grasp=`none`, contact=`push_surface`, motion=`push`, region=`button_top`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/push_buttons_episode161_51_61/combined_review.json`
- RoboPoint preferred contact points: `[[0.4, 0.6], [0.439, 0.598], [0.47, 0.6], [0.369, 0.602]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/push_buttons_episode161_51_61/view0_51.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/push_buttons_episode161_51_61/view1_51.png`

## put_groceries_in_cupboard_episode161_74_89

- Task: put the crackers in the cupboard
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_groceries_in_cupboard_episode161_74_89`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_groceries_in_cupboard_episode161_74_89/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['cupboard', 'box_like', 'hollow', 'shelf', 'open_container']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_groceries_in_cupboard_episode161_74_89/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`lift_and_place`, motion=`lift_then_insert`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/put_groceries_in_cupboard_episode161_74_89/combined_review.json`
- RoboPoint preferred contact points: `[[0.156, 0.6], [0.158, 0.55], [0.158, 0.65], [0.139, 0.51]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_groceries_in_cupboard_episode161_74_89/view0_74.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_groceries_in_cupboard_episode161_74_89/view1_74.png`

## put_money_in_safe_episode161_63_82

- Task: put the money away in the safe on the bottom shelf
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_money_in_safe_episode161_63_82`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_money_in_safe_episode161_63_82/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['safe', 'rectangular', 'hollow', 'slot', 'front_opening']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_money_in_safe_episode161_63_82/affordance_robopoint.json` (done)
- Affordance labels: grasp=`pinch`, contact=`insert_object`, motion=`insert`, region=`thin_object_edge`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/put_money_in_safe_episode161_63_82/combined_review.json`
- RoboPoint preferred contact points: `[[0.331, 0.04], [0.333, 0.1], [0.333, 0.15], [0.334, 0.2], [0.334, 0.242], [0.334, 0.0]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_money_in_safe_episode161_63_82/view0_63.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_money_in_safe_episode161_63_82/view1_63.png`

## place_shape_in_shape_sorter_episode161_70_85

- Task: put the cube in the shape sorter
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/place_shape_in_shape_sorter_episode161_70_85`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/place_shape_in_shape_sorter_episode161_70_85/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['shape_sorter', 'shape_profile', 'slot', 'hole', 'matching_geometry', 'alignment_sensitive']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/place_shape_in_shape_sorter_episode161_70_85/affordance_robopoint.json` (done)
- Affordance labels: grasp=`edge_grasp`, contact=`insert_object`, motion=`insert`, region=`shape_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/place_shape_in_shape_sorter_episode161_70_85/combined_review.json`
- RoboPoint preferred contact points: `[[0.3, 0.79], [0.259, 0.81], [0.23, 0.827], [0.269, 0.769], [0.331, 0.794], [0.231, 0.785], [0.3, 0.831]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/place_shape_in_shape_sorter_episode161_70_85/view0_70.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/place_shape_in_shape_sorter_episode161_70_85/view1_70.png`

## put_item_in_drawer_episode161_1_40

- Task: put the item in the bottom drawer
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_item_in_drawer_episode161_1_40`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_item_in_drawer_episode161_1_40/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['drawer', 'rectangular', 'hollow', 'open_container', 'sliding_axis', 'handle']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_item_in_drawer_episode161_1_40/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`lift_and_place`, motion=`lift_then_place`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/put_item_in_drawer_episode161_1_40/combined_review.json`
- RoboPoint preferred contact points: `[[0.486, 0.7], [0.539, 0.698], [0.58, 0.694], [0.517, 0.729], [0.508, 0.669]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_item_in_drawer_episode161_1_40/view0_1.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/put_item_in_drawer_episode161_1_40/view1_1.png`

## insert_onto_square_peg_episode161_63_78

- Task: put the ring on the red spoke
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/insert_onto_square_peg_episode161_63_78`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/insert_onto_square_peg_episode161_63_78/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['peg', 'square_peg', 'hole', 'alignment_sensitive', 'insertable_part']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/insert_onto_square_peg_episode161_63_78/affordance_robopoint.json` (done)
- Affordance labels: grasp=`edge_grasp`, contact=`insert_object`, motion=`insert`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/insert_onto_square_peg_episode161_63_78/combined_review.json`
- RoboPoint preferred contact points: `[[0.298, 0.61], [0.297, 0.652], [0.297, 0.569], [0.284, 0.529]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/insert_onto_square_peg_episode161_63_78/view0_63.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/insert_onto_square_peg_episode161_63_78/view1_63.png`

## open_drawer_episode161_62_76

- Task: open the bottom drawer
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/open_drawer_episode161_62_76`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/open_drawer_episode161_62_76/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['drawer', 'rectangular', 'hollow', 'handle', 'sliding_axis', 'front_face']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/open_drawer_episode161_62_76/affordance_robopoint.json` (done)
- Affordance labels: grasp=`handle_grasp`, contact=`pull_handle`, motion=`pull`, region=`handle`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/human_check_bundle/open_drawer_episode161_62_76/combined_review.json`
- RoboPoint preferred contact points: `[[0.1, 0.69], [0.102, 0.648], [0.131, 0.671], [0.133, 0.629]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/open_drawer_episode161_62_76/view0_62.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results_batch_02/demos/open_drawer_episode161_62_76/view1_62.png`
