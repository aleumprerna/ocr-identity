# OCR Models ka Difference — Hinglish Guide

Yeh guide `C:\OCR-New` ke current code aur official model documentation par based hai. Is project mein 6 OCR engines hain: **PaddleOCR, Tesseract, GLM-OCR, Mistral OCR, Unlimited-OCR aur AsterNet**.

**Aapke English + Hindi Aadhaar workflow ke liye PaddleOCR se start karna sabse practical hai**, kyunki isi route mein structured Aadhaar fields nikaalne ka code already hai. Yeh recommendation project integration par based hai; saare models ki accuracy ka measured winner abhi decide nahi hua hai.

## 1. Ek nazar mein comparison

Neeche wali capabilities **is project ke current adapter** ki hain. Original model mein extra features ho sakte hain jo `main.py` abhi expose nahi karta.

| Engine | Kaise chalta hai? | Hardware / service | Is project mein input | Output ka main use | Kab choose karein? |
|---|---|---|---|---|---|
| PaddleOCR | Text detect karke recognize karta hai | Local CPU par configured | Image | Text, confidence, boxes; Aadhaar fields bhi | English/Hindi ID images aur existing Aadhaar extraction |
| Tesseract | OCR engine + installed language data | Local CPU; native executable chahiye | Image | Text aur line confidence | Clean printed scans ka baseline / second opinion |
| GLM-OCR | Image dekhkar text generate karta hai | Ollama; CPU possible, GPU se acceleration mil sakta hai | Image | Generated text / possible Markdown | Local document recognition ko evaluate karna |
| Mistral OCR | Hosted document OCR API | Internet + API key; local GPU nahi chahiye | Image aur PDF | Page Markdown aur page metadata | PDF workflow aur cloud document processing |
| Unlimited-OCR | Generative document parsing model | Is adapter mein NVIDIA CUDA + BF16 required | Image | Text aur model ke layout tags | Compatible GPU par local document parsing evaluate karna |
| AsterNet | Line recognizer; CTC decoding | Local CPU via ONNX | Image; full document ya cropped line | Vietnamese/Latin text, confidence, boxes | Vietnamese text aur CPU deployment |

`--engine both` koi seventh model nahi hai. Yeh **PaddleOCR aur Tesseract** ko run karta hai; outputs alag rehte hain, automatically merge nahi hote.

## 2. OCR aur structured extraction ka difference

OCR ka kaam image se likha hua text padhna hai. Maan lo image mein yeh likha hai:

```text
Name: Demo Person
DOB: 01/01/2000
```

Structured extraction ka kaam us text ko fields mein rakhna hai:

```json
{
  "name": "Demo Person",
  "date_of_birth": "2000-01-01"
}
```

**Har engine ki `.json` file banne ka matlab yeh nahi ki har engine name, DOB aur address extract karta hai.** JSON kabhi sirf text aur metadata ka container hota hai.

Current project mein Aadhaar parser sirf PaddleOCR route par automatically apply hota hai. Ismein `parse_aadhaar()`, Hindi name recognition aur QR presence checking ka project code use hota hai. PaddleOCR model khud se yeh poora application schema provide nahi kar raha.

PAN aur voter ID ke liye separate structured field parser abhi nahi hai. PaddleOCR unka text read kar sakta hai, lekin Aadhaar schema se PAN/voter fields automatically nahi milenge. Aise documents ke raw text ke liye `--raw` use karein.

## 3. Har engine ko simple language mein samjhein

### PaddleOCR

Pehle image mein text regions locate hote hain, phir unka text recognize hota hai. Current code PP-OCRv5 use karta hai. Iske detection aur recognition components alag hain. [Official PP-OCRv5 documentation](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)

**Project mein kya useful hai:** CPU execution configured hai. Large English/Hindi scans par smaller mobile detector use hota hai. Structured Aadhaar mode English text ke saath targeted Hindi name recognition bhi karta hai. Raw Hindi OCR ke liye `--raw --lang hi` available hai.

**Limit:** OCR galat hua toh parser bhi galat field nikaal sakta hai. Address extraction printed order aur rules par depend karti hai. QR presence milne ka matlab QR payload decode ya document authenticity verify hona nahi hai.

**Meri recommendation:** Existing Aadhaar JSON workflow ka first choice. PAN/voter image ka text compare karne ke liye bhi baseline rakhein.

### Tesseract

Yeh local OCR engine hai. Python ka `pytesseract` package actual Tesseract executable ko call karta hai. Language ke hisaab se `.traineddata` files install honi chahiye. [Official Tesseract repository](https://github.com/tesseract-ocr/tesseract)

Clean, straight printed scans par isko baseline ke roop mein try karein. Tilt, noise, uneven background aur low resolution par preprocessing ki zaroorat pad sakti hai. [Image quality guide](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)

**Project mein kya useful hai:** English + Hindi ke liye `--lang eng+hin` de sakte hain, agar dono language packs installed hain. Text aur average word-based line confidence milti hai.

**Limit:** Current route Aadhaar fields parse nahi karta aur table structure reconstruct nahi karta. Sirf `pip install pytesseract` karne se native program install nahi hota.

**Meri recommendation:** PaddleOCR ke saath comparison aur clean printed pages ke liye useful second engine.

### GLM-OCR

Yeh multimodal document model hai: image se output text generate karta hai. Official model complex document understanding, tables aur formulas jaise tasks ke liye describe kiya gaya hai. [Official GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR)

**Project mein kya useful hai:** `glm_ocr.py` Ollama ke `glm-ocr:q8_0` ko call karta hai. Default Ollama address local hai. Image ka longest side 1600 pixels tak resize hota hai; generation limit 2048 output tokens hai.

**Limit:** Current route generic `Text Recognition:` prompt use karta hai. Official full layout pipeline aur specialized table/formula modes yahan implement nahi hue hain. Chhota text resizing se lose ho sakta hai; token limit par result truncate ho sakta hai. Hindi accuracy ko sample documents par separately verify karein.

**Meri recommendation:** Local generative OCR ka comparison candidate. CPU speed aur memory apni machine par measure karein; Aadhaar field JSON ke liye extra parser lagega.

### Mistral OCR

Yeh hosted API hai jo document images aur PDFs se page text/Markdown return karti hai. Official API mein document structure aur table-related capabilities bhi hain. [Official OCR documentation](https://docs.mistral.ai/studio/document-processing/basic_ocr)

**Project mein kya useful hai:** PNG, JPEG, WebP aur PDF accept hote hain. API key environment variable mein set hoti hai. Local GPU ki zaroorat nahi; selected document Mistral ke server ko send hota hai.

**Limit:** Internet, service access aur billing/quota par depend karta hai. Current adapter basic OCR request bhejta hai; custom Aadhaar annotations, advanced table options aur optional confidence settings enable nahi karta. `mistral-ocr-latest` alias ka underlying version change ho sakta hai.

**Meri recommendation:** Direct PDF processing chahiye aur cloud use acceptable hai toh pehle isko evaluate karein. Exact cost aur language accuracy actual usage se confirm karein; yahan fixed pricing ya best-accuracy claim nahi kiya gaya hai.

### Baidu Unlimited-OCR

Yeh document parsing model hai. Official example mein NVIDIA GPU par Transformers inference aur `gundam`/`base` image modes diye gaye hain. [Official Unlimited-OCR model card](https://huggingface.co/baidu/Unlimited-OCR)

**Project mein kya useful hai:** Local inference hota hai. `gundam` mode image crops use karta hai; `base` ek image configuration use karta hai. Default total token limit 32768 hai, jisme input bhi included hai.

**Limit:** Current adapter CUDA aur BF16 support check karta hai; CPU-only environment mein run nahi hoga. GPU available hone ke baad bhi sufficient VRAM chahiye. Setup ke liye separate environment recommended hai. Original model ke multipage examples hone ke bawajood, hamara adapter abhi one-image route use karta hai.

**Meri recommendation:** Compatible NVIDIA GPU available ho aur local layout parsing test karni ho tab evaluate karein. Sirf model bada hone se aapke Hindi/English cards par accuracy better hona prove nahi hota.

### AsterNet Vietnamese OCR

Yeh Vietnamese text-line recognition ke liye specialized model hai. Published artifacts mein ONNX weights aur vocabulary milte hain. [Official AsterNet model card](https://huggingface.co/asterxlabs/asternet-vietnamese-ocr)

**Project mein kya useful hai:** CPU par ONNX Runtime se chalta hai. `line` mode cropped line read karta hai. `document` mode mein RapidOCR sirf lines detect karta hai, aur AsterNet unka text read karta hai. API key ya Ollama nahi chahiye.

**Limit:** Published vocabulary mein Hindi/Devanagari nahi hai, isliye Hindi Aadhaar text ke liye suitable nahi hai. Current adapter greedy CTC decoding use karta hai; separate linguistic rescorer/beam search included nahi hai. Complex columns ka reading order approximate hai.

**Meri recommendation:** Vietnamese ke liye useful specialized candidate. English/Latin bhi try kar sakte hain, lekin Indian Hindi documents ke default engine ke roop mein choose na karein.

## 4. Mere use case ke liye kaunsa better hai?

Yeh choices current integration aur deployment requirements par based recommendations hain. Same dataset par all-model accuracy benchmark abhi nahi hua hai.

| Aapki requirement | Pehle kya try karein? | Reason |
|---|---|---|
| English/Hindi Aadhaar se name, DOB aur address JSON | **PaddleOCR** | Existing Aadhaar parser aur targeted Hindi name recognition integrated hain |
| PAN / voter image se raw text | **PaddleOCR + Tesseract comparison** | Existing CPU routes; actual field correctness compare kar sakte hain |
| Clean printed scan, simple local workflow | **Tesseract** | Native local OCR baseline; language packs configure karne honge |
| PDF directly process karni hai | **Mistral** | Current `main.py` mein direct PDF route isi engine ka hai |
| Local generative OCR, Ollama use karna hai | **GLM-OCR** | Existing Ollama integration available hai |
| Local document parsing, compatible NVIDIA GPU hai | **Unlimited-OCR ko evaluate karein** | Current adapter ka hardware requirement match hota hai |
| Vietnamese text, CPU par inference | **AsterNet** | Language-specific recognizer aur ONNX integration hai |
| Images cloud par send nahi karni | **Local PaddleOCR / Tesseract / AsterNet; local Ollama par GLM; GPU ho toh Unlimited** | Inference local rakha ja sakta hai; GLM ka configured host check karein |

## 5. Speed, cost aur accuracy ko kaise samjhein?

**Speed:** Abhi saare engines ka equal-condition speed test nahi hua. Image size, detected lines, CPU/GPU, model loading aur network time result badalte hain. AsterNet ki per-line latency ko kisi doosre model ki full-page latency se directly compare karna fair nahi hoga.

**Cost:** Local routes mein is code ke through hosted per-request OCR charge nahi lagta, lekin hardware, storage aur electricity ka cost hota hai. Mistral ka cloud usage account pricing/quota ke hisaab se hota hai. Model download aur dependency install ke liye local routes ko bhi initially internet chahiye ho sakta hai.

**Accuracy:** Ek model ke benchmark ka percentage doosre dataset ke percentage se directly compare mat karein. Aapke documents ki language, crop, font aur scan quality match honi chahiye. Generative models bhi missing, repeated ya extra text de sakte hain; unka output original image se verify karein.

**Confidence:** PaddleOCR, Tesseract aur AsterNet ke confidence numbers alag tareeke se calculate hote hain. `0.95` har engine mein same accuracy ka proof nahi hai. GLM aur Unlimited ke current outputs mein comparable line confidence nahi milti; Mistral ki optional confidence request bhi current adapter mein enable nahi hai.

## 6. Test commands

Har command us Python environment mein run karein jisme us engine ki dependencies installed hain. Setup links: [GLM](README_GLM.md), [Unlimited](README_UNLIMITED.md), [AsterNet](README_ASTERNET.md).

```powershell
cd C:\OCR-New

# PaddleOCR: default structured Aadhaar extraction
python main.py image.png --engine paddleocr

# PaddleOCR: raw English text, PAN/voter jaise documents ke comparison ke liye
python main.py pan.png --engine paddleocr --raw --lang en

# PaddleOCR: raw Hindi text
python main.py image.png --engine paddleocr --raw --lang hi

# Tesseract: English + Hindi packs installed hone chahiye
python main.py image.png --engine tesseract --lang eng+hin

# Dono local engines ka raw English text; separate outputs milenge
python main.py pan.png --engine both --raw --lang en --output output\compare-engines

# GLM: Ollama installed aur running hona chahiye
ollama pull glm-ocr:q8_0
python main.py image.png --engine glm

# Mistral: apni real API key set karein
$env:MISTRAL_API_KEY = "your_api_key"
python main.py image.png --engine mistral
python main.py document.pdf --engine mistral

# Unlimited: separate environment + compatible CUDA/BF16 GPU required
.\.venv-unlimited\Scripts\python.exe main.py image.png --engine unlimited

# AsterNet: already prepared project environment
.\.venv-asternet\Scripts\python.exe main.py image.png --engine asternet
.\.venv-asternet\Scripts\python.exe main.py output\asternet-smoke\vietnamese_line.png --engine asternet --asternet-mode line

# Automated integration/regression checks
python -m unittest discover -v
```

`document.pdf` example path hai; apni actual PDF ka path dein. Cropped Vietnamese sample previous smoke test mein generate kiya gaya tha. `--lang` PaddleOCR/Tesseract ke liye hai; is flag se AsterNet ya doosre models ki language capability change nahi hoti.

### Results kahan milenge?

| Engine | Default files |
|---|---|
| PaddleOCR | `output\<name>.txt`, `<name>.json`; structured mode mein `<name>.ocr.json` bhi |
| Tesseract | `output\tesseract\<name>.txt` aur `.json` |
| GLM-OCR | `output\glm\<name>.txt` aur `.json` |
| Mistral | `output\mistral\<name>.txt` aur `.json` |
| Unlimited-OCR | `output\unlimited\<name>.txt` aur `.json` |
| AsterNet | `output\asternet\<name>.txt` aur `.json` |

## 7. Abhi kya verify hua hai, aur kya pending hai?

Previous implementation verification mein **44 automated tests pass hue the**. Yeh routing, output saving, decoding, parsing aur error handling jaise behavior check karte hain. Test count ko model accuracy percentage mat samjhein; kai tests mocked responses use karte hain.

AsterNet ka real CPU inference `image.png` par complete hua aur 40 lines output mein aayi thi. Yeh line count hai, 40 correct lines ka accuracy claim nahi. Ek generated Vietnamese line ka recognized text expected text se exactly match hua tha. Ek synthetic line se real-world Vietnamese accuracy establish nahi hoti.

**Saare 6 engines ka same images aur manually verified reference text ke against complete accuracy/speed comparison abhi pending hai.** Mistral ke real API result aur Unlimited ke real GPU result ko yahan verified winner nahi bola gaya hai.

### Apne documents par winner kaise decide karein?

1. Same representative images choose karein: clear, blurry, Hindi, English, rotated aur mixed-layout samples.
2. Har image ka correct text aur required fields manually note karein.
3. Supported language wale engines se same images process karein. First model download/loading time aur later inference time alag note karein.
4. Name, DOB, ID digits aur address ki exact correctness compare karein. Aadhaar fields compare karte waqt doosre engines ke liye bhi equivalent extraction step lagega.
5. Raw OCR ke liye CER (character error rate) aur WER (word error rate) dekhein; dono mein lower better hota hai. Language aur punctuation normalization same rakhein.
6. Jo engine aapke required fields mein kam errors, acceptable runtime aur suitable deployment deta hai, usko select karein. Alag document types ke liye alag engine rakhna bhi practical ho sakta hai.

## 8. Code mapping

| CLI engine | Implementation |
|---|---|
| `paddleocr` | `main.py`: `extract_text()` aur `parse_aadhaar()` |
| `tesseract` | `main.py`: `extract_tesseract_text()` |
| `glm` / `glm-ocr` | `glm_ocr.py` |
| `mistral` | `mistral_ocr.py` |
| `unlimited` / `unlimited-ocr` | `unlimited_ocr.py` |
| `asternet` | `asternet_ocr.py` |
| `both` | `main.py` se PaddleOCR aur Tesseract ke separate calls |
