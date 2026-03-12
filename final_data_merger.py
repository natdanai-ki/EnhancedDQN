import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", default="data/raw/DataWeather2024.csv")
    parser.add_argument("--dust", default="data/raw/DustBoy1Year.csv")
    parser.add_argument("--out", default="data/processed/chiangmai_365d_hourly.csv")
    args = parser.parse_args()

    # NOTE: ตรงนี้เป็น template — ถ้า logic merge เดิมของอาจารย์ถูกต้องแล้ว
    # ให้ copy logic เดิมมาใส่แทนส่วนนี้ได้ทันที

    wdf = pd.read_csv(args.weather)
    ddf = pd.read_csv(args.dust)

    # === ตัวอย่างสมมติ: ต้องปรับให้ตรง format จริงของไฟล์ raw ของอาจารย์ ===
    # จุดประสงค์คือให้ได้ final columns:
    # day_of_year, hour, outdoor_temp, outdoor_humidity, pm10
    #
    # ถ้าอาจารย์มีตัว merge เดิมที่ใช้งานได้อยู่แล้ว ให้ “ใช้ตัวเดิม” และแค่เปลี่ยน path out ตามโครงนี้

    # placeholder minimal (ไม่ควรใช้แทนของจริงถ้า schema ไม่ตรง)
    # ----------------------------------------------------------
    # TODO: แทนที่ด้วย robust merge logic ของอาจารย์
    # ----------------------------------------------------------
    raise NotImplementedError(
        "ใช้ final_data_merger เดิมของอาจารย์ได้เลยถ้า chiangmai_365d_hourly.csv พร้อมใช้งานแล้ว "
        "และให้ copy ไฟล์ไปทั้งสองเครื่องเพื่อให้ผลเทียบกันได้"
    )


if __name__ == "__main__":
    main()
