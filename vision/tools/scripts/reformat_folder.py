import os
import shutil

customers_folder = "/media/fruitspec-lab/cam175/customers"
customers_new_folder = "/media/fruitspec-lab/cam175/customers_new"

def get_row_number(row):
    return row[1:-1]

def get_row_new_name(row):
    return f"row_{get_row_number(row)}"

def get_scan_number(row):
    return 1 if row.endswith("A") else 2

def get_new_format_path(customers_new_folder, customer, block, scan_date, row):
    scan_number = get_scan_number(row)
    new_row_name = get_row_new_name(row)
    new_path = os.path.join(customers_new_folder, customer, block, scan_date, new_row_name, str(scan_number))
    return new_path

def rename_files_in_folder(new_path, row):
    scan_number = get_scan_number(row)
    for file in os.listdir(new_path):
        file_path = os.path.join(new_path, file)
        file_split = file.split(".")
        if len(file_split) != 2:
            continue
        file_name_til_point, ending = file_split
        if file_name_til_point.endswith("_1") or file_name_til_point.endswith("_2"):
            new_name = file_name_til_point[:-2] + "." + ending
            new_file_path = os.path.join(new_path, new_name)
            os.rename(file_path, new_file_path)
        if file == f"Result_FSI_{scan_number}_slice_data.json":
            os.remove(file_path)
        if file == f"Result_FSI_{scan_number}_slice_data_{row}.json":
            slice_json_path = os.path.join(new_path, f"Result_FSI_{scan_number}_slice_data_{row}.json")
            slice_json_path_new = os.path.join(new_path, f"Result_FSI_slice_data.json")
            os.rename(slice_json_path, slice_json_path_new)


for customer in os.listdir(customers_folder):
    customer_path = os.path.join(customers_folder, customer)
    if not os.path.isdir(customer_path):
        continue
    for scan_date in os.listdir(customer_path):
        scan_date_path = os.path.join(customer_path, scan_date)
        if not os.path.isdir(scan_date_path):
            continue
        for block in os.listdir(scan_date_path):
            block_path = os.path.join(scan_date_path, block)
            if not os.path.isdir(block_path):
                continue
            for row in os.listdir(block_path):
                row_path = os.path.join(block_path, row)
                if not os.path.isdir(row_path):
                    continue
                new_path = get_new_format_path(customers_new_folder, customer, block, scan_date, row)
                try:
                    print("moving: ", row_path)
                    os.makedirs(new_path, exist_ok=True)
                    for file_name in os.listdir(row_path):
                        new_file_path = os.path.join(new_path, file_name)
                        os.rename(os.path.join(row_path, file_name), new_file_path)
                    print(f"Directory '{row_path}' moved to '{new_path}' successfully!")
                except shutil.Error as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Unexpected error occurred: {e}")

                rename_files_in_folder(new_path, row)

