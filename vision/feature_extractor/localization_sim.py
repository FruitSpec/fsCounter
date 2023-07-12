"""
this is a simulation for testing the locatlization logic
"""

tracker_results = pd.read_csv("/home/fruitspec-lab/PycharmProjects/foliage/counter/T_32/det/fake_tracker.csv")
all_frames = tracker_results["image_id"].unique()
first_frame = all_frames[0]
first_tracker_results = tracker_results[tracker_results['image_id'] == first_frame]

fruit_space = {int(row[1]["track_id"]): (row[1]["x"],row[1]["y"],row[1]["z"])
               for row in first_tracker_results.iterrows()}
fruits_keys = list(fruit_space.keys())
for fruit in fruits_keys:
    if np.isnan(fruit_space[fruit][0]):
        fruit_space.pop(fruit)

for i, frame_number in enumerate(all_frames):
    if i  == 0:
        continue
    print(f"fruit space: {frame_number}")
    boxes = tracker_results[tracker_results['image_id'] == frame_number]

    new_boxes, old_boxes = {}, {}
    boxes_w, boxes_h = np.array([]), np.array([])
    for row in boxes.iterrows():
        row = row[1]
        id = int(row["track_id"])
        if id not in fruit_space.keys():
            new_boxes[id] = (row["x"], row["y"], row["z"])
        else:
            old_boxes[id] = (row["x"], row["y"], row["z"])
    n_closest = 5
    z_old = np.array([box[2] for box in old_boxes.values()])
    old_keys = np.array(list(old_boxes.keys()))
    for id, box in new_boxes.items():
        box_z = box[2]
        if np.isnan(box_z):
            continue
        dist_vec = np.abs([box_z - z_old])[0]
        closest_zs = np.sort(dist_vec)[:n_closest]
        closest_key = old_keys[[dist in closest_zs for dist in dist_vec]]
        shifts = [np.array(old_boxes[close_key]) - np.array(fruit_space[close_key]) for close_key in closest_key]
        box_projection = tuple(np.array(box) - np.nanmedian(shifts, axis=0))
        fruit_space[id] = box_projection
fruit_space
centers = np.array(list(fruit_space.values()))

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(-centers[:, 2], centers[:, 0], centers[:, 1])
for i, label in enumerate(fruit_space.keys()):  # plot each point + it's index as text above
    ax.text(-centers[i, 2], centers[i, 0], centers[i, 1], '%s' % (str(label)), size=15, zorder=1,
            color='k')
ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')
ax.view_init(20, 20)
plt.show()