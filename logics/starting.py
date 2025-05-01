import cv2
import numpy as np
from pyzbar.pyzbar import decode
from aiogram.types import BufferedInputFile, Message
from aiogram import F
from io import BytesIO
import os

from handlers.lead.bot_instance import bot, dp
from utils.logger_util import logger

@dp.message(F.text.lower() == "/start")
async def start_handler(message: Message):
    await message.answer("Отправь изображение с QR кодом.")

@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    np_img = np.frombuffer(file_bytes.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    qr_codes = decode(img)
    if not qr_codes:
        await message.reply("QR код не найден.")
        return

    qr = qr_codes[0]
    x, y, w, h = qr.rect
    qr_img = img[y:y + h, x:x + w]

    target_bg_color = np.array([253, 253, 229], dtype=np.uint8)
    white_mask = cv2.inRange(qr_img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    qr_img[white_mask > 0] = target_bg_color

    base_img_path = os.path.join("images", "img.png")
    base_img = cv2.imread(base_img_path)

    if base_img is None:
        logger("Failed to load base image")
        return

    resized_qr = cv2.resize(qr_img, (165, 165))

    (h, w) = resized_qr.shape[:2]
    center = (w // 2, h // 2)
    angle = -2.6

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_qr = cv2.warpAffine(resized_qr, rotation_matrix, (new_w, new_h), borderValue=(253, 253, 229))

    blurred_qr = cv2.GaussianBlur(rotated_qr, (5, 5), sigmaX=1.15)

    bh, bw, _ = base_img.shape
    qr_h, qr_w, _ = blurred_qr.shape
    x_offset = 518
    y_offset = 476

    roi = base_img[y_offset:y_offset + qr_h, x_offset:x_offset + qr_w]
    blended = cv2.addWeighted(roi, 0.05, blurred_qr, 0.95, 0)
    base_img[y_offset:y_offset + qr_h, x_offset:x_offset + qr_w] = blended

    _, buffer = cv2.imencode('.png', base_img)
    output_io = BytesIO(buffer.tobytes())
    output_io.name = "result.png"
    output_file = BufferedInputFile(output_io.getvalue(), filename=output_io.name)

    await message.answer_photo(photo=output_file)
