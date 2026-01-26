import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext(
    r"C:\Users\htalbi\workspace\captcha-project\runs\20260119_115159\captcha.png",
    detail=1,
    paragraph=False,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
print(result)
