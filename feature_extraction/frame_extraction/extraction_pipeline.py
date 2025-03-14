import argparse
from extract_frames import FrameExtractionManager
from extract_frames import FrameExtractor
from extract_metadata import MetadataGenerator
import logging


logging.basicConfig(level=logging.INFO)

def run_extraction_pipeline():
    
    parser = argparse.ArgumentParser(description="Frame Extraction Pipeline")
    
    parser.add_argument("--video_folder", type=str, default="data/full_videos/",
    # parser.add_argument("--video_folder", type=str, default="data/test_data/test_videos/",
                        help="Folder containing surgical videos.")
    parser.add_argument("--timestamp_folder", type=str, default="data/video_timestamps/",
    # parser.add_argument("--timestamp_folder", type=str, default="data/test_data/test_timestamps/",
                        help="Folder containing phase timestamp files.")
    parser.add_argument("--output_folder", type=str, default="/vol/scratch/SoC/misc/2024/sc22jg/frames/",
    # parser.add_argument("--output_folder", type=str, default="data/test_data/test_frames/",
                        help="Folder to store extracted frames.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel processes for frame extraction.")
    parser.add_argument("--frame_skip", type=int, default=5,
                        help="Number of frames to skip between extractions.")
    parser.add_argument("--generate_metadata", action="store_true",
                        help="Generate metadata CSV after frame extraction.")
    args = parser.parse_args()
    
        
    # # Step 1: Extract Frames
    # logging.info("Starting Frame Extraction...\n")
    # manager = FrameExtractionManager(
    #     video_folder=args.video_folder,
    #     timestamp_folder=args.timestamp_folder,
    #     output_folder=args.output_folder,
    #     num_workers=args.num_workers,
    #     frame_skip=args.frame_skip
    # )
    # manager.run()
    # logging.info("Frame Extraction Completed.")

    # Step 2: Generate Metadata CSV 
    if args.generate_metadata:
        logging.info("Generating Metadata CSV...\n")
        metadata_generator = MetadataGenerator(frame_folder=args.output_folder)
        metadata_generator.extract_frame_metadata()
        logging.info("Metadata CSV Generated.\n")
        
        
if __name__ == "__main__":
    run_extraction_pipeline()