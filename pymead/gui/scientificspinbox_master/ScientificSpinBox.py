from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
import re

"""
This code has been adapted from: https://gist.github.com/jdreaver/0be2e44981159d0854f5
Changes made are support for PyQt5, localisation, better intermediate state detection and better stepping.
Some inspiration taken from: https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/widgets/SpinBox.py
"""

decimal_point = re.escape(QtCore.QLocale().decimalPoint())
int_exp_regex = r'(([+-]?\d+(' + decimal_point + r'\d*)?|' + decimal_point + r'\d+)([eE][+]?\d+)?)'
exp_int_re = re.compile(int_exp_regex)
partial_int_re = re.compile(r'([+-]?(\d?))')
valid_int_re = re.compile(r'([+-]?\d*)')
partial_pos_int_regex = re.compile(r'([+]?(\d?))')
partial_pos_int_re = re.compile(partial_pos_int_regex)
partial_float_regex = r'([+-]?((\d*' + decimal_point + r'?))?\d*)'
partial_float_re = re.compile(partial_float_regex)


class IntValidator(QtGui.QValidator):
	"""
	Validates integer inputs for ScientificSpinBox
	"""

	@staticmethod
	def valid_integer_string(string: str) -> bool:
		"""
		Checks if string makes a valid integer, keeping in mind locale dependent decimal separators.
		"""

		exp_count = string.count('e') + string.count('E')
		if exp_count == 1:
			match = exp_int_re.search(string)
		elif exp_count == 0:
			match = valid_int_re.search(string)
		else:
			return False
		return match.groups()[0] == string if match else False

	@staticmethod
	def intermediate_pos_integer_string(string: str) -> bool:
		"""
		Checks if string makes a valid partial positive integer, keeping in mind locale dependent decimal separators.
		"""

		# Normal notation.
		# Use regex to check if string is part of a valid positive float:
		match = partial_pos_int_re.search(string)
		return match.groups()[0] == string if match else False

	@staticmethod
	def intermediate_float_string(string: str) -> bool:
		"""
		Checks if string makes a valid partial float, keeping in mind locale dependent decimal separators.
		"""

		# Normal notation.
		# Use regex to check if string is part of a valid positive float:
		match = partial_float_re.search(string)
		return match.groups()[0] == string if match else False

	@staticmethod
	def intermediate_integer_string(string: str) -> bool:
		"""
		Checks if string makes a valid partial integer, keeping in mind locale dependent decimal separators.
		"""

		# Allow empty strings:
		if string == "":
			return True

		# Allow 0 or 1 combined instances of e and E:
		exp_count = string.count('e') + string.count('E')
		if exp_count > 1:
			return False

		if exp_count == 1:
			# Scientific notation.
			# Split string at exponential symbol and check that both substrings are valid:
			char = 'e' if string.count('e') > 0 else 'E'
			_, __ = string.split(char)
			if IntValidator.intermediate_float_string(_) and IntValidator.intermediate_pos_integer_string(__):
				return True
		else:
			# Normal notation.
			# Use regex to check if string is part of a valid int:
			match = partial_int_re.search(string)
			return match.groups()[0] == string if match else False

		return False

	def validate(self, string: str, position: int) -> QtGui.QValidator.State:
		"""
		Validates input string to see if it is a valid integer or partial integer.

		Notes
		-----
		Position is not used, but required because we are overriding an internal method.
		"""

		if self.valid_integer_string(string):
			return QtGui.QValidator.State.Acceptable
		if self.intermediate_integer_string(string):
			return QtGui.QValidator.State.Intermediate
		else:
			return QtGui.QValidator.State.Invalid

	def fixup(self, string: str) -> str:
		"""
		Fixes up input text to create a valid integer. Puts an empty string on failure.
		"""

		match = exp_int_re.search(string)
		return match.groups()[0] if match else ""


class ScientificSpinBox(QtWidgets.QSpinBox):
	"""
	Subclass of QSpinBox that allows for scientific notation and is locale independent.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# On 64 bit windows, QSpinBox maximum size is limited to 32 bit, so check for this here:
		self.validator = IntValidator()
		try:
			self.setRange(-int(10e16), int(10e16))
		except OverflowError:
			self.setRange(-int(2 ** 31 - 1), int(2 ** 31 - 1))

	def validate(self, string: str, position: int) -> (QtGui.QValidator.State, str, int):
		"""
		Returns the validity of the string, using a QtGui.QValidator object.

		Notes
		-----
		Return type depends on whether PySide or PyQt is being used.
		"""

		validity = self.validator.validate(string, position)

		# support 2 different PyQt APIs.
		if hasattr(QtCore, 'QString'):
			return validity, position
		else:
			return validity, string, position

	def fixup(self, string: str) -> str:
		"""
		Fixes up strings that are considered intermediate when spinbox focus is lost.
		"""

		return self.validator.fixup(string)

	def valueFromText(self, string: str) -> int:
		"""
		Turns user viewable string into an int.
		"""

		if string == "":
			return self.value()

		string = string.replace(QtCore.QLocale().decimalPoint(), ".", 1)
		# Cannot convert scientific notation strings into ints directly, so convert to float first.
		value = int(float(string))

		if value > self.maximum():
			value = self.maximum()
		if value < self.minimum():
			value = self.minimum()

		return value

	def textFromValue(self, value: int) -> str:
		"""
		Turns internal int value into user viewable string.
		"""

		string = "{:g}".format(value).replace("e+", "e").replace(".", QtCore.QLocale().decimalPoint(), 1)
		string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
		return string

	def stepBy(self, steps: int):
		"""
		Increases or decreases the value of the least significant digit by 'steps'.
		"""

		text = self.cleanText()

		groups = exp_int_re.search(text).groups()

		val, _, __ = groups[1].partition(QtCore.QLocale().decimalPoint())
		significance = len(groups[1])-1
		if groups[1] is not None and len(groups[1]) > 0 and groups[1][0] == '-':
			significance -= 1
		val = float(val) + steps * 10**significance

		new_string = "{:g}".format(val) + (groups[3] if groups[3] else "")
		self.lineEdit().setText(new_string)
