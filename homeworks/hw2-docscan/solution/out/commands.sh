# Запускать из каталога решения
python docscan.py noise sample_data/desk.jpg
python docscan.py noise sample_data/corner.jpg
python docscan.py noise sample_data/tiles.jpg
python docscan.py noise sample_data/plain.jpg
python docscan.py noise sample_data/nosheet.jpg
python docscan.py noise sample_data/desk.jpg --add-sigma 5
python docscan.py noise sample_data/desk.jpg --add-sigma 10
python docscan.py noise sample_data/desk.jpg --add-sigma 25
python docscan.py edges sample_data/desk.jpg -o out/desk_edges_default.png
python docscan.py edges sample_data/desk.jpg --close 0 -o out/desk_edges_noclose.png
python docscan.py edges sample_data/desk.jpg --canny 50,150 -o out/desk_edges_fixed.png
python docscan.py edges sample_data/corner.jpg -o out/corner_edges_default.png
python docscan.py edges sample_data/corner.jpg --close 0 -o out/corner_edges_noclose.png
python docscan.py edges sample_data/corner.jpg --canny 50,150 -o out/corner_edges_fixed.png
python docscan.py edges sample_data/tiles.jpg -o out/tiles_edges_default.png
python docscan.py edges sample_data/tiles.jpg --close 0 -o out/tiles_edges_noclose.png
python docscan.py edges sample_data/tiles.jpg --canny 50,150 -o out/tiles_edges_fixed.png
python docscan.py edges sample_data/plain.jpg -o out/plain_edges_default.png
python docscan.py edges sample_data/plain.jpg --close 0 -o out/plain_edges_noclose.png
python docscan.py edges sample_data/plain.jpg --canny 50,150 -o out/plain_edges_fixed.png
python docscan.py eval sample_data --ann sample_data/corners.json --seeds 20 --csv out/eval_default.csv --overlays out/eval_default
python docscan.py eval sample_data --ann sample_data/corners.json --seeds 20 --canny 50,150 --csv out/eval_fixed.csv --overlays out/eval_fixed
python docscan.py eval sample_data --ann sample_data/corners.json --seeds 20 --close 0 --csv out/eval_noclose.csv --overlays out/eval_noclose
python docscan.py scan sample_data/desk.jpg --canny 50,150 --binarize otsu -o out/desk_scan.jpg --bw out/desk_otsu.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/desk.jpg --canny 50,150 --binarize adaptive -o out/desk_scan.jpg --bw out/desk_adaptive.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/desk.jpg --canny 50,150 --binarize flat -o out/desk_scan.jpg --bw out/desk_flat.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/corner.jpg --canny 50,150 --binarize otsu -o out/corner_scan.jpg --bw out/corner_otsu.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/corner.jpg --canny 50,150 --binarize adaptive -o out/corner_scan.jpg --bw out/corner_adaptive.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/corner.jpg --canny 50,150 --binarize flat -o out/corner_scan.jpg --bw out/corner_flat.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/tiles.jpg --canny 50,150 --binarize otsu -o out/tiles_scan.jpg --bw out/tiles_otsu.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/tiles.jpg --canny 50,150 --binarize adaptive -o out/tiles_scan.jpg --bw out/tiles_adaptive.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/tiles.jpg --canny 50,150 --binarize flat -o out/tiles_scan.jpg --bw out/tiles_flat.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/plain.jpg --canny 50,150 --binarize otsu -o out/plain_scan.jpg --bw out/plain_otsu.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/plain.jpg --canny 50,150 --binarize adaptive -o out/plain_scan.jpg --bw out/plain_adaptive.png --gt sample_data/page-gt.png
python docscan.py scan sample_data/plain.jpg --canny 50,150 --binarize flat -o out/plain_scan.jpg --bw out/plain_flat.png --gt sample_data/page-gt.png
python docscan.py video --source sample_data/clip.mp4 --method contour --no-window --seconds 0
python docscan.py video --source sample_data/clip.mp4 --method hough --no-window --seconds 0
python docscan.py video --source sample_data/clip.mp4 --method auto --no-window --seconds 0 --save out/video.mp4
