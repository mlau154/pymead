from pyqtgraph import TableWidget
import benedict


def flatten_dict(jmea_dict: dict):
    dben = benedict.benedict(jmea_dict)
    keypaths = dben.keypaths()

    # data = {"name": [], "lower": [], "value": [], "upper": [], "active": [], "eq": []}
    data = []

    for k in keypaths:
        p = dben[k]
        # if isinstance(p, dict) and "_value" in p.keys():
        #     if isinstance(p["_value"], list):
        #         data["name"].extend([k, ""])  # Add a spacer to account for multiple values applying to one name
        #         data["value"].extend(p["_value"])
        #         data["lower"].extend([a[0] for a in p["bounds"]])
        #         data["upper"].extend([a[1] for a in p["bounds"]])
        #         data["active"].extend(p["active"])
        #         data["eq"].extend([a.strip("{}") for a in p["func_str"].split(",")])
        #     else:
        #         data["name"].append(k)  # Add a spacer to account for multiple values applying to one name
        #         data["value"].append(p["_value"])
        #         data["lower"].append(p["bounds"][0])
        #         data["upper"].append(p["bounds"][1])
        #         data["active"].append(p["active"])
        #         data["eq"].append(p["func_str"])
        if isinstance(p, dict) and "_value" in p.keys():
            if isinstance(p["_value"], list):
                eq1 = "" if p["func_str"] is None else p["func_str"].split(",")[0].strip("{}")
                eq2 = "" if p["func_str"] is None else p["func_str"].split(",")[1].strip("{}")
                data1 = {"name": k, "value": p["_value"][0], "lower": p["bounds"][0][0], "upper": p["bounds"][0][1],
                         "active": p["active"][0], "eq": eq1}
                data2 = {"name": "", "value": p["_value"][1], "lower": p["bounds"][1][0], "upper": p["bounds"][1][1],
                         "active": p["active"][1], "eq": eq2}
                data.extend([data1, data2])
            else:
                eq = "" if p["func_str"] is None else p["func_str"]
                data1 = {"name": k, "value": p["_value"], "lower": p["bounds"][0], "upper": p["bounds"][1],
                         "active": p["active"], "eq": eq}
                data.append(data1)

    return data


class BoundsValuesTable(TableWidget):
    def __init__(self, jmea_dict: dict):
        data = flatten_dict(jmea_dict)
        super().__init__(editable=True, sortable=True)
        self.setData(data)
