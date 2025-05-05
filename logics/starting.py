import cv2
import numpy as np
from pyzbar.pyzbar import decode
from aiogram.types import BufferedInputFile, Message
from aiogram import F
from io import BytesIO
import os

from handlers.lead.bot_instance import bot, dp
from utils.logger_util import logger

def rotate_image_with_transparency(image, angle):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def add_noise_to_center_area(image, sigma=50, area_ratio=0.6):
    h, w = image.shape[:2]
    c = image.shape[2]
    cx, cy = w // 2, h // 2
    half_w = int(w * area_ratio / 2)
    half_h = int(h * area_ratio / 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = 255
    mask_multi = cv2.merge([mask] * c)

    noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16)
    noisy_image[mask_multi == 255] += noise[mask_multi == 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def darken_image(image, factor=0.85):
    darkened = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return darkened

def process_qr_image(img, padding=5):
    qr_codes = decode(img)
    if not qr_codes:
        return None, None
    qr = qr_codes[0]
    x, y, w, h = qr.rect
    x_start = max(x - padding, 0)
    y_start = max(y - padding, 0)
    x_end = min(x + w + padding, img.shape[1])
    y_end = min(y + h + padding, img.shape[0])
    qr_img = img[y_start:y_end, x_start:x_end]
    return img, qr_img


@dp.message(F.text.lower() == "/start")
async def start_handler(message: Message):
    await message.answer("Отправь изображение с QR кодом.")

@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    np_img = np.frombuffer(file_bytes.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed_img, qr_img = process_qr_image(img)
    if processed_img is None:
        await message.reply("QR код не найден.")
        return

    base_img_path = os.path.join("images", "img.png")
    base_img = cv2.imread(base_img_path)
    if base_img is None:
        logger("Failed to load base image")
        return

    qr_with_noise = add_noise_to_center_area(qr_img, sigma=50, area_ratio=0.9)
    qr_with_noise = darken_image(qr_with_noise, factor=0.98)

    rotated_qr = rotate_image_with_transparency(qr_with_noise, -2)
    resized_qr = cv2.resize(rotated_qr, (213, 213))

    resized_qr_bgr = cv2.cvtColor(resized_qr, cv2.COLOR_BGRA2BGR)

    x_offset, y_offset = 607, 570
    h, w = resized_qr_bgr.shape[:2]
    base_img[y_offset:y_offset + h, x_offset:x_offset + w] = resized_qr_bgr

    _, buffer_main = cv2.imencode('.png', base_img)
    main_output = BufferedInputFile(BytesIO(buffer_main.tobytes()).getvalue(), filename="result.png")
    await message.answer_photo(photo=main_output)

    _, buffer_qr = cv2.imencode('.png', qr_img)
    qr_output = BufferedInputFile(BytesIO(buffer_qr.tobytes()).getvalue(), filename="qr_only.png")
    await message.answer_document(document=qr_output)
