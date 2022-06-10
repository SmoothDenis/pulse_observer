import os
import time

import cv2
import dlib
import numpy as np
from scipy import signal
from scipy.fft import fft

# Константы для работы программы
MIN_HZ = 0.83  # 50 BPM - минимальная допустимая частота сердечных сокращений
MAX_HZ = 4.0  # 240 BPM - максимально допустимая частота сердечных сокращений
MIN_DATA = 100  # Минимальное количество кадров, требуемых до расчета частоты сердечных сокращений.
# Более высокие значения = долгое ожидание, но большая точность.
TITLE_APP = "Main APP"
BUFFER_SIZE = 1000  # Количество недавних средних значений для расчёта
MAX_VALUES_TO_GRAPH = 30  # Количество недавних средних значений для графика пульса
DEBUG_MODE = True  # Включить для отладки
# DEBUG_MODE = False

average_bpm = []
average_bpm_print = []
pulse_to_GUI = 0
fft_list = []
freqs_list = []


def rtsp_init():
    RTSP_LINK = "rtsp://192.168.3.45:8554/mjpeg/1"
    try:
        RTSP_LINK = os.environ["RTSP_LINK"]
    except:
        print("RTSP_LINK not found in environment variables")
    print(RTSP_LINK)


# Получает область, которая включает лоб, глаза и нос.
def get_fullface_region(face_landmark):
    array_points = np.zeros((len(face_landmark.parts()), 2))
    for i, part in enumerate(face_landmark.parts()):
        array_points[i] = (part.x, part.y)

    left, right, top, bottom = save_only_face(array_points)
    return int(left), int(right), int(top), int(bottom)


def save_only_face(array_landmark):
    # Сохраняются только те точки, которые соответствуют внутренним чертам лица (например, рот, нос, глаза, брови)
    # Подробнее:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(np.min(array_landmark[17:47, 0]))
    min_y = int(np.min(array_landmark[17:47, 1]))
    max_x = int(np.max(array_landmark[17:47, 0]))
    max_y = int(np.max(array_landmark[17:47, 1]))

    center = min_x + (max_x - min_x) / 2
    left = min_x + int((center - min_x) * 0.15)
    right = max_x - int((max_x - center) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return left, right, top, bottom


# Выделение области лба
def get_forehead_regions(face_landmarks):
    # Хранение в Numpy массиве для более простой работы с данными
    array_landmarks = np.zeros((len(face_landmarks.parts()), 2))
    for i, part in enumerate(face_landmarks.parts()):
        array_landmarks[i] = (part.x, part.y)

    # Область лба между бровями
    # Подробнее:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(array_landmarks[21, 0])
    min_y = int(min(array_landmarks[21, 1], array_landmarks[22, 1]))
    max_x = int(array_landmarks[22, 0])
    max_y = int(max(array_landmarks[21, 1], array_landmarks[22, 1]))
    left = min_x - 10
    right = max_x + 10
    top = min_y - (max_x - min_x) - 10
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)


# Получает область носа.
def get_nose_region(face_landmarks):
    array_landmarks = np.zeros((len(face_landmarks.parts()), 2))
    for i, part in enumerate(face_landmarks.parts()):
        array_landmarks[i] = (part.x, part.y)

    # Нос и щёки
    # Подробнее:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(array_landmarks[36, 0])
    min_y = int(array_landmarks[28, 1])
    max_x = int(array_landmarks[45, 0])
    max_y = int(array_landmarks[33, 1])
    left = min_x
    right = max_x
    # top = min_y + (min_y * 0.02)
    # bottom = max_y + (max_y * 0.02)
    top = min_y + (min_y * 0.01)
    bottom = max_y * 0.99
    return int(left), int(right), int(top), int(bottom)


def filter_clear_signal_data(data, fps):
    # Проверка, что массив не имеет бесконечных или пустых значений
    data = np.array(data)
    np.nan_to_num(data, copy=False)

    # Линейное сглаживание сигнала
    a_trended = signal.detrend(data, type="linear")
    a_meaned = adaptive_filter_anomalies(a_trended, 15)
    # Фильтрация сигнала с помощью полосового фильтра Баттерворта
    filtered = butterworth_bandpass_filter(a_meaned, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered


# Среднее значение для интересующих регионов. Также нарисует зеленый прямоугольник вокруг
def get_regions_avg_from_points(frame, view, face_landmarks, draw_rect=True):
    # Получает координаты регионов
    fh_left, fh_right, fh_top, fh_bottom = get_forehead_regions(face_landmarks)
    nose_left, nose_right, nose_top, nose_bottom = get_nose_region(face_landmarks)

    # Рисует рамки вокруг регионов, когда включен режим отладки
    if draw_rect:
        cv2.rectangle(
            view,
            (fh_left, fh_top),
            (fh_right, fh_bottom),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.rectangle(
            view,
            (nose_left, nose_top),
            (nose_right, nose_bottom),
            color=(0, 255, 0),
            thickness=2,
        )

    # Усреднение значений в областях интересующих нас
    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
    nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
    return get_avg_regions(fh_roi, nose_roi)


def adaptive_filter_anomalies(signal_values, num_windows):
    limits_slice = int(round(len(signal_values) / num_windows))
    zip_slice = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), limits_slice):
        if i + limits_slice > len(signal_values):
            limits_slice = len(signal_values) - i
        curr_slice = signal_values[i : i + limits_slice]
        if DEBUG_MODE and curr_slice.size == 0:
            print(
                "Empty Slice: size={0}, i={1}, window_size={2}".format(
                    signal_values.size, i, limits_slice
                )
            )
            print(curr_slice)
        zip_slice[i : i + limits_slice] = curr_slice - np.mean(curr_slice)
    return zip_slice


# Создает фильтр Баттерворта и применяет его
# Подробнее:  http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butterworth_bandpass_filter(data, low, high, frame_rate, order=5):
    frame_rate = 10
    filter_rate = frame_rate * 0.5
    # print(data, low, high, sample_rate, order)
    low /= filter_rate
    high /= filter_rate
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.lfilter(b, a, data)


# Усредняет значения зеленых полей для двух массивов пикселей
def get_avg_regions(roi1, roi2):
    region1_green = roi1[:, :, 1]
    region2_green = roi2[:, :, 1]
    avg = (np.mean(region1_green) + np.mean(region2_green)) / 2.0
    return avg


# Возвращает абсолютный максимум из массива
def get_max_abs(lst):
    return max(max(lst), -min(lst))


# Рисует график пульса когда включен режим отладки
def graph_GUI(signal_values, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_x = float(graph_width) / MAX_VALUES_TO_GRAPH

    autoscale_vert(signal_values, graph_height, graph, scale_x)
    return graph


def autoscale_vert(signal_values, graph_height, graph, scale_factor_x):
    # Автоматическое масштабирование по вертикали на основе значения с наибольшим абсолютным значением
    max_abs = get_max_abs(signal_values)
    scale_y = (float(graph_height) / 2.0) / max_abs

    middle_y = graph_height / 2
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(middle_y + signal_values[i] * scale_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(middle_y + signal_values[i + 1] * scale_y)
        cv2.line(
            graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1
        )


# Отображение количества кадров в секунду
def fps_GUI(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        "FPS: " + str(round(fps, 2)),
        (5, 20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(0, 255, 0),
    )
    return frame


# Отображение текста в окне программы
def graph_GUI_text(text, color, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    text_size, text_base = cv2.getTextSize(
        text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1
    )
    text_x = int((graph_width - text_size[0]) / 2)
    text_y = int((graph_height / 2 + text_base))
    cv2.putText(
        graph,
        text,
        (text_x, text_y),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        color=color,
        thickness=1,
    )
    return graph


# Отображает текст частоты сердечных сокращений
def bpm_GUI(bpm_str, bpm_width, bpm_height):
    if pulse_to_GUI != 0:
        bpm_str = str(pulse_to_GUI)
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    bpm_text_size, bpm_text_base = cv2.getTextSize(
        bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7, thickness=2
    )
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height / 2 + bpm_text_base)
    cv2.putText(
        bpm_display,
        bpm_str,
        (bpm_text_x, bpm_text_y),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=2.7,
        color=(0, 255, 0),
        thickness=2,
    )
    bpm_label_size, bpm_label_base = cv2.getTextSize(
        "BPM", fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, thickness=1
    )
    bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
    bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    cv2.putText(
        bpm_display,
        "BPM",
        (bpm_label_x, bpm_label_y),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.6,
        color=(0, 255, 0),
        thickness=1,
    )
    return bpm_display


# Функция для расчёта пульса
def compute_filter_bpm(filtered_values, fps, buffer_size, last_bpm):
    global average_bpm
    global average_bpm_print
    global pulse_to_GUI
    global fft_list
    global freqs_list
    # Расчёт фильтра частот на основе быстрого преобразования Фурье
    fft = np.abs(np.fft.rfft(filtered_values))

    # Генерация списка частот, соответствующих значениям БПФ
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)

    if DEBUG_MODE == True:
        save_data_to_file(fft, freqs)
    # Фильтрация любых аномалий в БПФ, которые не входят в диапазон [MIN_HZ, MAX_HZ]
    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps > MAX_HZ:
            # print('BPM of {0} deleted.'.format(bps * 60.0))
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break

    # Частота сердечных сокращений не может меняться более чем на 10%
    # (на самом деле для 15 кадров в секунду - 20%) между выборками,
    # поэтому используется средневзвешенное значение, чтобы сгладить значение пульса
    # В данном случае для баланса выбраны значения 65% прошлого и 35% текущего значения пульса
    if last_bpm > 0:
        bpm = (last_bpm * 0.65) + (bpm * 0.35)
        average_bpm.append(bpm)
        if len(average_bpm) > 5:
            average_bpm_print.append(np.median(average_bpm))
            average_bpm = []
        if len(average_bpm_print) > 5:
            pulse_to_GUI = int(np.median(average_bpm_print))
            print(int(np.median(average_bpm_print)))
            # print('FFT: ', fft_list)
            # print('FREQS: ', freqs_list)
            average_bpm_print = []
    return bpm


def save_data_to_file(fft, freqs):
    global average_bpm
    global average_bpm_print
    global pulse_to_GUI
    global fft_list
    global freqs_list
    fft_list = []
    freqs_list = []
    fft_list.append(fft)
    freqs_list.append(freqs)
    with open("fft_list", "w") as file:
        file.write(str(fft_list))
    with open("freqs_list", "w") as file:
        file.write(str(freqs_list))


# Гравная функция для запуска программы
def pulse_detector_set_var(detector, predictor, webcam, window):
    regions_avg_values = []
    graph_values = []
    times = []
    last_bpm = 0
    graph_height = 200
    graph_size_width = 0
    bpm_display_width = 0

    # cv2.getWindowProperty() returns -1 when window is closed by user.
    # while cv2.getWindowProperty(window, 0) == 0:
    while True:
        ret_val, frame = webcam.read()

        # ret_val == False if unable to read from webcam
        if not ret_val:
            print(
                "ERROR:  Unable to read from webcam.  Was the webcam disconnected?  Exiting."
            )
            end_program(webcam)

        # Копия кадра для отображения на экране, оригинал будет использоваться для расчета пульса
        view = np.array(frame)

        # 75% отводится под пульс, остальные - под график
        if graph_size_width == 0:
            graph_size_width = int(view.shape[1] * 0.75)
            if DEBUG_MODE:
                print("Graph width = {0}".format(graph_size_width))
        if bpm_display_width == 0:
            bpm_display_width = view.shape[1] - graph_size_width

        face_accomodation(
            detector,
            predictor,
            webcam,
            window,
            regions_avg_values,
            graph_values,
            times,
            last_bpm,
            graph_height,
            graph_size_width,
            bpm_display_width,
            frame,
            view,
        )


def face_accomodation(
    detector,
    predictor,
    webcam,
    window,
    regions_avg_values,
    graph_values,
    times,
    last_bpm,
    graph_height,
    graph_width,
    bpm_display_width,
    frame,
    view,
):
    # Определение положения лица в кадре при помощи DLIB
    faces = detector(frame, 0)
    if len(faces) == 1:
        face_points = predictor(frame, faces[0])
        roi_avg = get_regions_avg_from_points(frame, view, face_points, draw_rect=True)
        regions_avg_values.append(roi_avg)
        times.append(time.time())

        # Когда буфер заполнен, удаление первого элемента из буфера
        if len(times) > BUFFER_SIZE:
            regions_avg_values.pop(0)
            times.pop(0)

        curr_buffer_size = len(times)

        # Пока в буфере недостаточно данных, не производится расчет пульса
        if curr_buffer_size > MIN_DATA:
            # Вычисление оставшегося времени
            time_elapsed = times[-1] - times[0]
            fps = curr_buffer_size / time_elapsed  # кадры в секунду
            # Очистка данных сигнала
            filtered = filter_clear_signal_data(regions_avg_values, fps)

            graph_values.append(filtered[-1])
            if len(graph_values) > MAX_VALUES_TO_GRAPH:
                graph_values.pop(0)

                # Вывести график пульса
            graph = graph_GUI(graph_values, graph_width, graph_height)
            # Высчитать и отобразить пульс
            bpm = compute_filter_bpm(filtered, fps, curr_buffer_size, last_bpm)
            bpm_display = bpm_GUI(str(int(round(bpm))), bpm_display_width, graph_height)
            last_bpm = bpm
            # Отобразить частоту кадров
            if DEBUG_MODE:
                view = fps_GUI(view, fps)

        else:
            # Если не получилось вычислить пульс, график будет пустым
            pct = int(round(float(curr_buffer_size) / MIN_DATA * 100.0))
            loading_text = "Computing pulse: " + str(pct) + "%"
            graph = graph_GUI_text(loading_text, (0, 255, 0), graph_width, graph_height)
            bpm_display = bpm_GUI("--", bpm_display_width, graph_height)

    else:
        # Нет обнаруженных лиц, сбросить данные
        del regions_avg_values[:]
        del times[:]
        graph = graph_GUI_text(
            "No face detected", (0, 0, 255), graph_width, graph_height
        )
        bpm_display = bpm_GUI("--", bpm_display_width, graph_height)

    graph = np.hstack((graph, bpm_display))
    view = np.vstack((view, graph))

    manage_imshow(webcam, window, view)


def manage_imshow(webcam, window, view):
    if DEBUG_MODE == True:
        cv2.imshow(window, view)

    key = cv2.waitKey(1)
    # При нажатии клавиши выйти
    if key == 27:
        end_program(webcam)


# Закрытие окон и завершение программы
def end_program(webcam):
    if DEBUG_MODE == True:
        webcam.release()
    cv2.destroyAllWindows()
    exit(0)


def main():
    detector = dlib.get_frontal_face_detector()
    # Предварительно обученная модель распознавания может быть загружена из:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

    try:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except RuntimeError as e:
        print(
            "ERROR: 'shape_predictor_68_face_landmarks.dat' was not found in current directory."
        )
        return

    rtsp_init()
    # webcam = cv2.VideoCapture(RTSP_LINK)
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print(
            "ERROR: Unable to open webcam. Verify connection and try again."
        )
        webcam.release()
        return

    if DEBUG_MODE == True:
        cv2.namedWindow(TITLE_APP)
    pulse_detector_set_var(detector, predictor, webcam, TITLE_APP)

    end_program(webcam)


if __name__ == "__main__":
    main()
