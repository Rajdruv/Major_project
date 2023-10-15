import csv
import pandas as pd


def convert_to_csv(log_file, export_csv_file):
    with open(log_file, "r+", encoding="utf-8") as read_file:
        data = read_file.read()
    time_filter = data.split("# Time")
    all_data = []
    for each_record in time_filter:
        if "Rows_examined" in each_record:
            try:
                all_rows = each_record.split("#")
                time = all_rows[0].replace("\n", "").replace(": ", "")
                user_host = all_rows[1].split("Id:")[0].replace("User@Host:", "")
                id = all_rows[1].split("Id:")[1].strip()
                other_data = all_rows[2].split()
                query_time = other_data[1]
                lock_time = other_data[3]
                rows_spent = other_data[5]
                rows_examined = other_data[7]
                query = " ".join(other_data[8:])
                temp = {
                    "time": time,
                    "user_host": user_host,
                    "id": id,
                    "query_time": query_time,
                    "lock_time": lock_time,
                    "rows_spent": rows_spent,
                    "rows_examined": rows_examined,
                    "query": query,
                }
                all_data.append(temp)
            except:
                print("DATA: {}".format(each_record))
    df = pd.DataFrame(all_data)
    df.to_csv(export_csv_file, escapechar="\\")


def convert_main():
    convert_to_csv("bikram.log", "bikram_log.csv")
    convert_to_csv("dhruv-sep18.log", "dhruv_log.csv")
    convert_to_csv("gaurav-uptosep23.log", "gaurav_log.csv")
    convert_to_csv("sujith-uptoSep18.log", "sujith_log.csv")


if __name__ == "__main__":
    convert_main()
