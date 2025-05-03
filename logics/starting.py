import cv2
import numpy as np
from pyzbar.pyzbar import decode
from aiogram.types import BufferedInputFile, Message
from aiogram import F
from io import BytesIO
import os

from handlers.lead.bot_instance import bot, dp
from utils.logger_util import logger

def recolor_white_background(image, white_thresh=245, bg_color=(254, 189, 123)):
    lower_white = np.array([white_thresh, white_thresh, white_thresh], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(image, lower_white, upper_white)
    recolored = image.copy()
    recolored[white_mask == 255] = bg_color
    return recolored


def replace_specific_background(img, target_color=(252, 253, 226), new_color=(250, 226, 134), tolerance=30):
    img_copy = img.copy()
    lower = np.array([max(0, c - tolerance) for c in target_color], dtype=np.uint8)
    upper = np.array([min(255, c + tolerance) for c in target_color], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    img_copy[mask == 255] = new_color
    return img_copy

def process_qr_image(img):
    img = recolor_white_background(img, white_thresh=200, bg_color=(252, 253, 226))
    img = replace_specific_background(img, target_color=(252, 253, 226), new_color=(250, 226, 134))
    qr_codes = decode(img)
    if not qr_codes:
        return None, None
    qr = qr_codes[0]
    x, y, w, h = qr.rect
    qr_img = img[y:y + h, x:x + w]
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
    resized_qr = cv2.resize(qr_img, (170, 170))
    x_offset, y_offset = 624, 588
    h, w = resized_qr.shape[:2]
    bg_patch = processed_img[y_offset:y_offset + h, x_offset:x_offset + w].copy()
    combined_patch = bg_patch.copy()
    combined_patch[0:h, 0:w] = resized_qr
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -0.99, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    rotated_patch = cv2.warpAffine(combined_patch, rotation_matrix, (new_w, new_h), borderValue=(250, 226, 134))
    blurred_patch = cv2.GaussianBlur(rotated_patch, (9, 9), sigmaX=0.5)
    roi = base_img[y_offset:y_offset + blurred_patch.shape[0], x_offset:x_offset + blurred_patch.shape[1]]
    base_img[y_offset:y_offset + blurred_patch.shape[0], x_offset:x_offset + blurred_patch.shape[1]] = cv2.addWeighted(roi, 0.05, blurred_patch, 0.95, 0)
    _, buffer_main = cv2.imencode('.png', base_img)
    main_output = BufferedInputFile(BytesIO(buffer_main.tobytes()).getvalue(), filename="result.png")
    await message.answer_photo(photo=main_output)
    _, buffer_qr = cv2.imencode('.png', qr_img)
    qr_output = BufferedInputFile(BytesIO(buffer_qr.tobytes()).getvalue(), filename="qr_only.png")
    await message.answer_document(document=qr_output)
