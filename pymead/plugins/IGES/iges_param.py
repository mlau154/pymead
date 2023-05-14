from datetime import datetime


class IGESParam:
    def __init__(self, value, dtype: str):
        self.value = value
        self.dtype = dtype
        allowed_dtypes = ["string", "int", "real", "datetime", "none"]
        if self.dtype == "datetime" and not isinstance(self.value, datetime):
            raise TypeError(f"datetime was selected as the dtype for IGESParam with value {self.value}, but "
                            f"the type was {type(self.value)}. 'value' must be of type datetime.datetime.")
        if self.dtype not in allowed_dtypes:
            raise ValueError(f"GlobalParam dtype must be one of {allowed_dtypes}. Chosen value was {self.dtype}.")

    def write_value_to_python_str(self):
        if self.dtype == "int":
            return str(int(self.value))
        elif self.dtype == "real":
            return f"{self.value:.12E}"
        elif self.dtype == "string":
            return f"{len(self.value)}H{self.value}"  # Hollerith format string
        elif self.dtype == "datetime":
            return f"15H{self.value.strftime('%Y%m%d.%H%M%S')}"
        elif self.dtype == "none":
            return ""
