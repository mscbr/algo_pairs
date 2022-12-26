def truncate_float(number, decimals):
    float_str = str(number)
    float_arr = float_str.split(".")
    float_arr[1] = float_arr[1][:decimals]
    return float(".".join(float_arr))
