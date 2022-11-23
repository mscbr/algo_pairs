
def get_pos_neg_color(number):
    if number > 0:
        return "[32m"
    elif number < 0:
        return "[31m"
    else:
        return "[0m"
