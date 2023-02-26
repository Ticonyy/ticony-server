import pytesseract
import cv2
import pyzbar.pyzbar as pyzbar
import re
import base64
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
# from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/api')
def hello_world():
    #config = ('-l kor --oem 3 --psm 4')

    config = ('-l kor+eng --oem 3 --psm 4')

    #img_path = "./chupachups_test.jpg"
    img_path = "./chupachups_test.jpg"

    img_gray = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resizing
    width = 1000
    aspect_ratio = float(width) / img_gray.shape[1]
    dsize = (width, int(img_gray.shape[0] * aspect_ratio))
    resized_gray = cv2.resize(img_gray, dsize, interpolation=cv2.INTER_AREA)

    # grayscale -> binary
    binary_gray = cv2.threshold(resized_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    gray_text = pytesseract.image_to_string(binary_gray, config=config)

    return gray_text

@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        ################################ image 가져오는 방법 ############################
        # f = request.files['file']
        # f.save(secure_filename(f.filename))
        # print(f.filename)

        # img_path = "./" + f.filename
        # print(img_path)


        # img_gray = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        # img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ################################################################################
        config = ('-l kor+eng --oem 3 --psm 4')

        d = dict(code='default', name='default', place='default', year='efault', month='default', day='default')

        base64image = request.json[0]['image']
        print(base64image)
        imageStr = base64.b64decode(base64image)
        nparr = np.fromstring(imageStr, np.uint8)

        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        # if img_gray:
        #     print("image exists.")


        for code in pyzbar.decode(img_gray):
            barcodenum = code.data.decode('utf-8')
            d["code"] = str(barcodenum)
        # Resizing
        width = 1000
        aspect_ratio = float(width) / img_gray.shape[1]
        dsize = (width, int(img_gray.shape[0] * aspect_ratio))
        resized_gray = cv2.resize(img_gray, dsize, interpolation=cv2.INTER_AREA)

        # grayscale -> binary
        binary_gray = cv2.threshold(resized_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        oem = 2
        # psm = 0
        output = pytesseract.image_to_string(binary_gray, config=config)
        print("hi")
        print(output)
        # return output
        ############################################
        ############################################

        ############################################유효기간################################################

        indexGigan = output.find("기한")
        if indexGigan == -1:
            indexGigan = output.find("기간")

        candidates = output[indexGigan+2:]
        # print(candidates)

        answer = ""
        for i in candidates:
            if i.isnumeric():
                answer = answer + str(i)
        # print(answer)

        year = answer[:4]
        month = answer[4:6]
        day = answer[6:8]
        d["year"] = year
        d["month"] = month
        d["day"] = day

        ############################################교환처################################################

        indexGyo = output.find("교")
        indexCho = output.find("처")
        indexSa = output.find("사")

        candidates2=d["place"]
        # 교환처의 경우
        if (indexGyo != -1 and indexCho != -1):
            candidates2 = output[indexCho+1:]
            indexYoo = candidates2.find("유")
            indexSang = candidates2.find("상")

            if indexSang != -1 and indexYoo > indexSang:
                candidates2 = candidates2[:indexSang].strip()
            elif indexSang == -1:
                candidates2 = candidates2[:indexYoo].strip()
        # 사용처의 경우
        elif (indexSa != -1 and indexCho != -1):
            candidates2 = output[indexCho+1:]
            indexGifticon = candidates2.find("gifticon")
            if indexGifticon != -1:
                candidates2 = candidates2[:indexGifticon].strip()
                # print(indexGifticon)
            else:
                indexMoon = candidates2.find("문")
                candidates2 = candidates2[:indexMoon].strip()


        candidates2 = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s.]", "", candidates2)
        candidates2 = re.sub(r"[\n]", "", candidates2)
        regex = re.compile(r'\d\d\d\d \d')
        matchobj = regex.search(candidates2)
        if matchobj is not None:
            found = matchobj.group()
            indexLast = candidates2.find(found)
            candidates2 = candidates2[:indexLast]

        d["place"] = candidates2.strip()
        if len(d["place"]) >= 40:
            d["place"] = "default"       
        #############################################상품명###############################################
        candidates3=d["name"]
        indexSang = output.find("상")
        indexMyeong = output.find("명")
        indexShot = output.find("Shot")
        indexGifticon2 = output.find("gifticon")
        indexTing = output.find("Ting")
        #상품명이 언급되었을 경우
        if indexTing != -1:
            indexGyo = output.find("교")
            regex = re.compile(r'\d\d\d\d \d')
            matchobj = regex.search(output)
            found = matchobj.group()
            indexLast = output.find(found)
            candidates3 = output[indexLast+19:indexGyo]
        elif (indexSang != -1 and indexMyeong != -1):
            candidates3 = output[indexMyeong+1:]
            indexGyo = candidates3.find("교")
            indexSooryang = candidates3.find("수량")
            if (indexSooryang != -1 or indexGyo != -1):
                if indexGyo == -1 :
                    tmp=indexSooryang
                elif indexSooryang == -1 :
                    tmp=indexGyo
                candidates3 = candidates3[:tmp].strip()
        elif indexShot == -1 and indexGifticon2 != -1:
            indexRyang = output.find("량")
            candidates3 = output[indexGifticon2+8:indexRyang-1]
        elif indexShot == -1 and indexGifticon2 == -1:
            regex = re.compile(r'\d\d\d\d \d')
            matchobj = regex.search(output)
            found = matchobj.group()
            indexLast = output.find(found)
            candidates3 = output[:indexLast]
        elif indexShot != -1:
            regex = re.compile(r'\d\d\d\d-\d')
            matchobj = regex.search(output)
            found = matchobj.group()
            indexLast = output.find(found)
            candidates3 = output[indexShot+4:indexLast]

        candidates3 = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s.()]", "", candidates3)
        candidates3 = re.sub(r"[\n]", "", candidates3)
        candidates3 = ' '.join(candidates3.split())


        candidates3 = candidates3.strip()
        indexGyo = candidates3.find("교")
        if indexGyo != -1:
            candidates3 = candidates3[:indexGyo]
        d["name"] = candidates3
        if len(d["name"]) >= 40:
            d["name"] = "default"
        ############################################
        ############################################


        return jsonify(d)
    else:
        return render_template('file_upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)