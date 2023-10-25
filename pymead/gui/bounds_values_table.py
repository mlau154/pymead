from copy import deepcopy

from pyqtgraph import TableWidget
import benedict


def flatten_dict(jmea_dict: dict):
    dben = benedict.benedict(jmea_dict)
    keypaths = dben.keypaths()

    data = []

    for k in keypaths:
        p = dben[k]
        if isinstance(p, dict) and "_value" in p.keys():
            if isinstance(p["_value"], list):
                eq1 = "" if p["func_str"] is None else p["func_str"].split(",")[0].strip("{}")
                eq2 = "" if p["func_str"] is None else p["func_str"].split(",")[1].strip("{}")
                data1 = {"name": k, "value": p["_value"][0], "lower": p["bounds"][0][0], "upper": p["bounds"][0][1],
                         "active": int(bool(p["active"][0])), "eq": eq1}
                data2 = {"name": "", "value": p["_value"][1], "lower": p["bounds"][1][0], "upper": p["bounds"][1][1],
                         "active": int(bool(p["active"][1])), "eq": eq2}
                data.extend([data1, data2])
            else:
                eq = "" if p["func_str"] is None else p["func_str"]
                data1 = {"name": k, "value": p["_value"], "lower": p["bounds"][0], "upper": p["bounds"][1],
                         "active": int(bool(p["active"])), "eq": eq}
                data.append(data1)

    return data


class BoundsValuesTable(TableWidget):
    def __init__(self, jmea_dict: dict):
        data = flatten_dict(jmea_dict)
        super().__init__(editable=True, sortable=False)
        self.setData(data)
        self.original_data = deepcopy(self.data())

    def data(self):
        rows = list(range(self.rowCount()))
        columns = list(range(self.columnCount()))
        data = []
        if self.horizontalHeadersSet:
            row = []
            if self.verticalHeadersSet:
                row.append('')

            for c in columns:
                row.append(self.horizontalHeaderItem(c).text())
            data.append(row)

        for r in rows:
            row = []
            if self.verticalHeadersSet:
                row.append(self.verticalHeaderItem(r).text())
            for c in columns:
                item = self.item(r, c)
                if item is not None and item.value != "":
                    row.append(item.value)
                else:
                    row.append(None)
            data.append(row)

        return data

    def compare_data(self):
        original_data = self.original_data

        new_data = self.data()

        # flattened_new = flatten_dict(self.data())

        indices_to_modify = []
        params_to_modify = []
        data_to_modify = {}

        for idx, (row_original, row_new) in enumerate(zip(original_data, new_data)):
            if row_original != row_new:
                indices_to_modify.append(idx)

        for idx in indices_to_modify:
            if new_data[idx][0] == "" and new_data[idx - 1][0] not in params_to_modify:
                params_to_modify.append(new_data[idx - 1][0])
            else:
                params_to_modify.append(new_data[idx][0])
            if idx + 1 < len(new_data):
                if new_data[idx + 1][0] == "" and new_data[idx + 1][0] not in params_to_modify:
                    params_to_modify.append(new_data[idx + 1][0])

        for idx in indices_to_modify:
            nd = new_data[idx]
            # Treat the positional parameters (PosParam) differently than the regular parameters (Param)
            if nd[0] is None:
                # Identified that this row in the table corresponds to the "y" value of a PosParam
                ndy = nd
                ndx = new_data[idx - 1]
            elif (idx + 1 < len(new_data)) and new_data[idx + 1][0] is None:
                # Identified that the next row in the table corresponds to the "y" value of a PosParam
                ndx = nd
                ndy = new_data[idx + 1]
            else:
                ndx, ndy = None, None

            if ndx is None and ndy is None:
                data_to_modify[nd[0]] = {"value": nd[1], "bounds": [nd[2], nd[3]], "active": bool(nd[4]), "eq": nd[5]}
            else:
                data_to_modify[ndx[0]] = {
                    "value": [ndx[1], ndy[1]],
                    "bounds": [[ndx[2], ndx[3]], [ndy[2], ndy[3]]],
                    "active": [bool(ndx[4]), bool(ndy[4])],
                    "eq": ndx[5]
                }

        # TODO: make equation cell wider

        return data_to_modify
