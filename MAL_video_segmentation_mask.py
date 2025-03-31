import sys
import datetime

import labelbox as lb
import labelbox.types as lb_types
import nanoid
import imageio.v3 as iio
import ndjson

from utils import generate_composite_mask_from_instances, LabelboxClassInstance

VIDEO_URL = "https://avtshare01.rz.tu-ilmenau.de/avt-vqdb-uhd-1/test_1/segments/bigbuck_bunny_8bit_200kbps_360p_60.0fps_h264.mp4"
API_KEY = None
ANNOTATION_PROJECT = "mal-video-segmentation-masks-issue"
MAL_START_FRAME = 1
MAL_END_FRAME = 20
MAL_SKIP_FRAME = 2

if __name__ == "__main__":
    API_KEY = sys.argv[1] if len(sys.argv) > 1 else API_KEY
    if not API_KEY:
        raise ValueError("You need to provide the labelbox api key (with admin role)!")

    client = lb.Client(API_KEY)

    # create dataset and 1 dummy datarow
    print("Creating dataset and datarow")
    dataset = client.create_dataset(name="video-test")
    global_key = f"video-{nanoid.generate(size=5)}"
    datarow = dataset.create_data_row(
        row_data=VIDEO_URL, 
        global_key=global_key)


    # TODO Create ontology
    print("Creating demo ontology")
    ontology_builder = lb.OntologyBuilder(
        tools=[
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="bunny"),
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="tree"),
            lb.Tool(tool=lb.Tool.Type.RASTER_SEGMENTATION, name="butterfly"),
        ]
    )
    ontology = client.create_ontology(
        "VideoMaskSegmentation Demo",
        ontology_builder.asdict(),
        media_type=lb.MediaType.Video
    )

    print("Creating annotation project and add batch")
    project = client.create_project(
        name=ANNOTATION_PROJECT,
        description="",
        media_type=lb.MediaType.Video
    )
    project.connect_ontology(ontology)

    task = project.create_batches(
        name_prefix="batch-",
        global_keys=global_key,
    )

    print("Errors: ", task.errors())
    print("Result: ", task.result())

    ### MAL PAYLOAD MASKS
    print("Creating MAL payload")
    

    class_instances = [
        LabelboxClassInstance(class_name="bunny", class_idx=1, rgb=(255,0,0)),
        LabelboxClassInstance(class_name="tree",  class_idx=2, rgb=(0,255,0)),
        LabelboxClassInstance(class_name="tree",  class_idx=2, rgb=(0,255,100)),
        LabelboxClassInstance(class_name="butterfly", class_idx=3, rgb=(0,0,255)),
    ]

    n_frames, height, width = iio.improps(VIDEO_URL).shape[:3]
    MAL_END_FRAME = len(n_frames) + 1 if not MAL_END_FRAME else MAL_END_FRAME
    
    mask_frames = []
    instances = []

    # create masks
    frame_indices = list(range(MAL_START_FRAME, MAL_END_FRAME, MAL_SKIP_FRAME))
    for frame_idx in frame_indices:
        
        # create a fake composite mask that randomly change on each frame
        # composite mask will have len(class_instances) square colored with LabelboxClassInstance.rgb value 
        composite_mask = generate_composite_mask_from_instances(width, height, class_instances, seed=frame_idx)
        composite_mask_bytes = iio.imwrite("<bytes>", composite_mask, extension=".png") 
        mask_frames.append(
            lb_types.MaskFrame(
                index=frame_idx, 
                im_bytes=composite_mask_bytes
            )
        )
    # create instances mapping
    for instance in class_instances:
        instances.append(
            lb_types.MaskInstance(color_rgb=instance.rgb, name=instance.class_name)
        )

    # create a label for datarow
    label = lb_types.Label(
        data= {"global_key": global_key},
        annotations = [
            lb_types.VideoMaskAnnotation(frames=mask_frames, instances=instances)
        ]
    )


    # MAL Import job
    print("Importing MAL annotations")
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        predictions = [label],
        name=f"mal-{nanoid.generate()}"
    )

    upload_job.wait_till_done()

    print(f"Errors: {upload_job.errors}")
    print(f"Status of uploads: {upload_job.statuses}")

    # Manual Work
    print("*"*50)
    input("You need to manually move datarow from initial labeling status. When done, press enter to continue...")
    print("*"*50
    )
    # Project Export Job
    export_task = project.export(params={})

    export_task.wait_till_done()

    print(export_task.result)


    timestamp = datetime.datetime.now().isoformat().split(".")[0]
    filename = f"export_{project.name}_{timestamp}.ndjson"
    datarows = [datarow.json for datarow in export_task.get_buffered_stream()]

    with open(filename, 'w') as f:
        ndjson.dump(datarows, f)

    print(f"Exported data saved into {filename}")