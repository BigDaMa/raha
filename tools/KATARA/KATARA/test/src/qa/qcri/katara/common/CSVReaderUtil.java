package qa.qcri.katara.common;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.log4j.Logger;

import qa.qcri.katara.kbcommon.util.Pair;
import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;
import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

public class CSVReaderUtil {

	private final static Logger logger = Logger.getLogger(CSVReaderUtil.class);

	static class IntegerWrapper {
		private int value;

		public void setValue(int value) {
			this.value = value;
		}

		public int getValue() {
			return value;
		}
	}

	private static String[] toStringArray(Map<String, String> map) {
		String[] retValue = new String[map.size()];
		Map<String, String> treeMap = new TreeMap<String, String>(map);
		int i = 0;
		for (String s : treeMap.keySet()) {
			retValue[i] = treeMap.get(s);
			i++;
		}
		return retValue;
	}
	
	public static void writeToCSV(List<Map<String, String>> list, String filePath) throws IOException {
		CSVWriter writer = new CSVWriter(new  FileWriter(filePath));
		for (Map<String, String> m : list) {
			String[] strArray = toStringArray(m);
			writer.writeNext(strArray);
		}
		writer.close();
	}
	
	public static int getNumOfCol(final String filePath) {
		CSV csv = CSV.create();
		CSVReader reader = csv.reader(filePath);
		String[] columns = null;
		try {
			columns = reader.readNext();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return columns.length;
	}

	public static List<String> getCellsInRowBySampling(final String filePath,
			final int colNum, final int sample) {
		final List<String> cells = new ArrayList<String>();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex % sample == 0)
					cells.add(values[colNum]);
			}
		});
		return cells;
	}

	public static List<String> getCellsInRowBySampling(final String filePath,
			final String col, final int sample) {
		final List<String> cells = new ArrayList<String>();
		final IntegerWrapper wrapper = new IntegerWrapper();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex == 0) {
					int count = 0;
					for (String val : values) {
						if (val.equals(col)) {
							wrapper.setValue(count);
						}
						count++;
					}
				} else {
					String val = values[wrapper.getValue()];
					if (rowIndex % sample == 0)
						cells.add(val);
				}
			}
		});
		return cells;
	}

	/**
	 * call this function to get all the cells when the first line of the csv is
	 * not the header name
	 * 
	 * @param filePath
	 *            full path of the file
	 * @param colNum
	 *            the number of column number
	 * @return the list of the cells found in the specific column
	 */
	public static List<String> getCellsInRow(final String filePath,
			final int colNum) {
		final List<String> cells = new ArrayList<String>();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (colNum < values.length)
					cells.add(values[colNum]);
				else {
					cells.add("");
					logger.warn(filePath + "rowIndex" + rowIndex
							+ "don't have column" + colNum
							+ "use empty string instead");
				}
			}
		});
		return cells;
	}

	/**
	 * call this function to get all the cells when the first line of the csv is
	 * the header name
	 * 
	 * @param filePath
	 *            full path of the file
	 * @param colNum
	 *            the number of column number
	 * @return the list of the cells found in the specific column
	 */
	public static List<String> getCellsInRow(final String filePath,
			final String col) {
		final List<String> cells = new ArrayList<String>();
		final IntegerWrapper wrapper = new IntegerWrapper();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex == 0) {
					int count = 0;
					for (String val : values) {
						if (val.equals(col)) {
							wrapper.setValue(count);
						}
						count++;
					}
				} else {
					String val = values[wrapper.getValue()];
					cells.add(val);
				}
			}
		});
		return cells;
	}

	public static List<Map<String, String>> getAllCellsWithHeader(
			final String filePath) {
		final List<Map<String, String>> retValue = new ArrayList<Map<String, String>>();
		CSV csv = CSV.create();
		final Map<Integer, String> columnsName = new HashMap<Integer, String>();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex == 0) {
					int count = 0;
					for (String val : values) {
						columnsName.put(count, val);
						count++;
					}
				} else {
					Map<String, String> tuple = new HashMap<String, String>();
					int count = 0;
					for (String val : values) {
						String columnName = columnsName.get(count);
						tuple.put(columnName, val);
						count++;
					}
					retValue.add(tuple);
				}
			}
		});
		
		return retValue;
	}

	public static List<Pair<String, String>> getPairsInRowBySampling(
			final String filePath, final int colNum1, final int colNum2,
			final int sample) {
		final List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex % sample == 0)
					pairs.add(new Pair<String, String>(values[colNum1],
							values[colNum2]));
			}
		});
		return pairs;
	}

	public static List<Pair<String, String>> getPairsInRowBySampling(
			final String filePath, final String col1, final String col2,
			final int sample) {
		final List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		final IntegerWrapper wrapper1 = new IntegerWrapper();
		final IntegerWrapper wrapper2 = new IntegerWrapper();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex == 0) {
					int count = 0;
					for (String val : values) {
						if (val.equals(col1)) {
							wrapper1.setValue(count);
						}
						if (val.equals(col2)) {
							wrapper2.setValue(count);
						}
						count++;
					}
				} else {
					Pair<String, String> pair = new Pair<String, String>(
							values[wrapper1.getValue()], values[wrapper2
									.getValue()]);
					if (rowIndex % sample == 0)
						pairs.add(pair);
				}
			}
		});
		return pairs;
	}

	public static List<Pair<String, String>> getPairsInRow(
			final String filePath, final int colNum1, final int colNum2) {
		final List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				Pair<String, String> pair = new Pair<String, String>(
						values[colNum1], values[colNum2]);
				pairs.add(pair);
			}
		});
		return pairs;
	}

	public static List<Pair<String, String>> getPairsInRow(
			final String filePath, final String col1, final String col2) {
		final List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		final IntegerWrapper wrapperFirst = new IntegerWrapper();
		final IntegerWrapper wrapperSecond = new IntegerWrapper();
		CSV csv = CSV.create();
		csv.read(filePath, new CSVReadProc() {
			@Override
			public void procRow(int rowIndex, String... values) {
				if (rowIndex == 0) {
					int count = 0;
					for (String val : values) {
						if (val.equals(col1)) {
							wrapperFirst.setValue(count);
						}
						if (val.equals(col2)) {
							wrapperSecond.setValue(count);
						}
						count++;
					}
				} else {
					Pair<String, String> pair = new Pair<String, String>(
							values[wrapperFirst.getValue()],
							values[wrapperSecond.getValue()]);
					pairs.add(pair);
				}
			}
		});
		return pairs;
	}

}
