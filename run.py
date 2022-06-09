import subprocess
import glob
from src.yolo_process import *
from wand.image import Image
from tqdm import tqdm
from numberOCR.numberorc import *
import cv2


def sharpen_images(input_path: str, output_path):
    if input_path[-1] != '/':
        input_path = input_path + "/"

    if output_path[-1] != '/':
        output_path = output_path + "/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    kernel = np.array(([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), dtype=float)
    files = os.listdir(input_path)
    for f in tqdm(files):
        im = cv2.imread(input_path + f)
        im = cv2.filter2D(im, -1, kernel)
        cv2.imwrite(output_path + f, im)


def skew_correction(input_path: str, output_path):
    if input_path[-1] != '/':
        input_path = input_path + "/"

    if output_path[-1] != '/':
        output_path = output_path + "/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(input_path)
    for f in tqdm(files):
        if os.path.exists(output_path + f):
            continue

        with Image(filename=input_path + f) as img:
            img.deskew(0.4 * img.quantum_range)
            img.save(filename=output_path + f)


def detect_answers(input_path, output_path, temp_folder='temp', darknet_path='darknet'):
    if input_path[-1] != '/':
        input_path = input_path + "/"

    list_file = open(os.path.join(temp_folder, "answer_files.txt"), 'w')

    for pathAndFilename in glob.iglob(os.path.join(input_path, "*.JPG")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        list_file.write("../" + input_path + title + ext + "\n")

    list_file.close()

    cmd = "{darknet} detector test {data} {config} {weight} -dont_show -ext_output -out {output} < {input}".format(
        darknet="darknet.exe",
        data="data/answer.data",
        config="cfg/yolov4-answer.cfg",
        weight="trained/260422/yolov4-answer_last.weights",
        output=os.path.join("../", output_path),
        input=os.path.join("../", temp_folder, "answer_files.txt")
    )

    subprocess.check_call(cmd, shell=True, cwd=darknet_path)


def detect_numbers(input_path, output_path, threshold=0.99998, checkpoint='numberOCR/checkpoint/cp.ckpt'):
    results_df = pd.DataFrame(columns=['Filename', 'MD', 'SBD', 'MD_conf', 'SBD_conf', 'Check'])
    predictor = load_nnet_model(checkpoint)

    list_files = os.listdir(input_path)
    for filename in tqdm(list_files):
        img = cv2.imread(os.path.join(input_path, filename))
        result = extract(predictor, img)
        result['Filename'] = filename
        result['Check'] = (result['MD_conf'] < threshold) or (result['SBD_conf'] < threshold)

        results_df = results_df.append(result, ignore_index=True)

    results_df.to_csv(output_path)


if __name__ == '__main__':
    img_path = "all/answer-18-4"
    temp_folder = "temp/answer-18-4/"
    skew_correction_folder = os.path.join(temp_folder, "skew_correction")

    yolo_answers_path = temp_folder + "yolo_answers.txt"
    numbers_path = os.path.join(temp_folder, "numbers.csv")

    answers_path = os.path.join(temp_folder, "result.xlsx")

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # sharpen_images(os.path.join(img_path, "answer"), os.path.join(img_path, "sharpened"))
    #skew_correction(os.path.join(img_path, "answer"), skew_correction_folder)
    #detect_numbers(os.path.join(img_path, "number"), numbers_path)
    detect_answers(skew_correction_folder,  yolo_answers_path, temp_folder)
    results_to_xlsx(yolo_answers_path, numbers_path, answers_path, skew_correction_folder, save_img=True, save_img_path=temp_folder + "answer_imgs")
