/**
 * Author: Xu Chu
 */
package qa.qcri.katara.dbcommon;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import jline.internal.InputStreamReader;

import org.apache.jena.atlas.lib.StrUtils;

import au.com.bytecode.opencsv.CSVReader;

public class Table implements Serializable {

	public String inputDBPath;
	private ColumnMapping colMap;
	private String schema;

	private int numRows;
	private int numCols;
	private List<Tuple> tuples = new ArrayList<Tuple>();

	public String tableName;

	public Table(String inputDBPath, int numRows) {
		this.inputDBPath = inputDBPath;
		this.numRows = numRows;

		if (inputDBPath.endsWith(".xml"))
			initFromSunitaXMLFile();
		else
			initFromCSVFile();

		tableName = inputDBPath.substring(inputDBPath
				.lastIndexOf(File.separatorChar + "") + 1);

		System.out.println("Table initialized: " + colMap.getColumnHead()
				+ " Num Rows: " + this.getNumRows());
	}

	public Table(Table table) {
		this.inputDBPath = table.inputDBPath;
		this.colMap = new ColumnMapping(table.colMap.getColumnHead());
		this.schema = table.schema;
		this.numRows = 0;
		this.numCols = table.numCols;
		this.tableName = table.tableName;
		for (Tuple tuple : table.tuples) {
			tuples.add(new Tuple(tuple));
		}
	}

	private boolean isInteger(String s) {
		try {
			Integer.parseInt(s);
		} catch (NumberFormatException e) {
			return false;
		}
		// only got here if we didn't return false
		return true;
	}

	private boolean isDouble(String s) {
		try {
			Double.parseDouble(s);
		} catch (NumberFormatException e) {
			return false;
		}
		// only got here if we didn't return false
		return true;
	}

	public String getFilePath() {
		return inputDBPath;
	}

	private void initFromCSVFile() {
		int temp = 0;
		try {
			CSVReader csvReader = new CSVReader(new InputStreamReader(new FileInputStream(inputDBPath), "UTF-8"),
					',', '"');

			String[] colValues;
			while ((colValues = csvReader.readNext()) != null) {
				if (temp == 0) {
					// First line is the columns information
					colMap = new ColumnMapping(StrUtils.strjoin(",", colValues));
					String[] columns = colValues;
					this.numCols = columns.length;
					temp++;
				} else {
					boolean nullColumn = false;
					// NULl column checking
					for (String value : colValues) {
						if (value == null || value.equals("")
								|| value.equals("?") || value.contains("?")
								|| value.equalsIgnoreCase("null")) {
							//System.out.println("NULL columns");
							nullColumn = true;
							break;
						}
					}
					
					//xuchu: don't skip any rows because of null column
					 //if(nullColumn) 
					 	//{ continue; }
					 

					if (colValues.length != colMap.getColumnNames().length) {
						continue;
					}

					Tuple tuple = new Tuple(colValues, colMap, temp - 1);
					tuples.add(tuple);
					temp++;
					if (temp > numRows) {
						break;
					}
				}
			}
			numRows = tuples.size();
			System.out.println("NumRows:  " + numRows + " NumCols:" + numCols);

			csvReader.close();
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
	}

	private void initFromSunitaXMLFile() {

		String content = null;
		File file = new File(inputDBPath); // for ex foo.txt
		try {
			FileReader reader = new FileReader(file);
			char[] chars = new char[(int) file.length()];
			reader.read(chars);
			content = new String(chars);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int i1 = content.indexOf("<header>");
		int i2 = content.indexOf("</header>");
		String header = content.substring(i1, i2 + 9);
		String temp = null;
		ArrayList<String> colValues = new ArrayList<String>();
		while (header.indexOf("<text>") != -1
				&& header.indexOf("</text>") != -1) {
			temp = header.substring(header.indexOf("<text>"),
					header.indexOf("</text>") + 7);
			String col = temp.substring(6, temp.length() - 7);
			colValues.add(col);
			header = header.substring(header.indexOf("</text>") + 7);
		}
		colMap = new ColumnMapping(StrUtils.strjoin(",", colValues));
		this.numCols = colValues.size();

		String body = content.substring(content.indexOf(header));
		int numCols = 0;
		int tid = 1;
		colValues.clear();
		while (body.indexOf("<text>") != -1 && body.indexOf("</text>") != -1
				&& body.indexOf("</row>") != -1) {
			temp = body.substring(body.indexOf("<text>"),
					body.indexOf("</text>") + 7);
			String col = temp.substring(6, temp.length() - 7);
			colValues.add(col);
			if (colValues.size() == this.numCols) {
				Tuple t = new Tuple(colValues.toArray(new String[0]), colMap,
						tid);
				tuples.add(t);
				tid++;
				colValues.clear();
			}

			body = body.substring(body.indexOf("</text>") + 7);
		}
		this.numRows = tuples.size();

	}

	private void initFromFile() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inputDBPath));
			String line = null;
			int temp = 0;
			while ((line = br.readLine()) != null) {
				if (temp == 0) {
					// First line is the columns information
					colMap = new ColumnMapping(line);
					schema = line;
					String[] columns = line.split(",");
					this.numCols = columns.length;
					temp++;
				} else {
					String[] colValues = line.split(",");
					// Length checking
					if (colValues.length != this.numCols) {
						// System.out.println("Input Database Error");
						continue;
					}
					boolean nullColumn = false;
					// NULl column checking
					for (String value : colValues) {
						if (value.equals("") || value.equals("?")
								|| value.contains("?")) {
							// System.out.println("NULL columns");
							nullColumn = true;
							break;
						}
					}
					if (nullColumn) {
						continue;
					}

					// Type checking
					boolean tag = true;
					for (int i = 0; i < numCols; i++) {
						String coliValue = colValues[i];

						String type = colMap.positionToType(i + 1);
						if (type.equalsIgnoreCase("Integer")
								&& !isInteger(coliValue)) {
							tag = false;
							break;
						} else if (type.equalsIgnoreCase("Double")
								&& !isDouble(coliValue)) {
							tag = false;
							break;
						}
					}
					if (tag == false)
						continue;
					// TODO adjust to 0-index
					Tuple tuple = new Tuple(colValues, colMap, temp - 1);
					tuples.add(tuple);
					temp++;
					if (temp > numRows) {
						break;
					}

				}
			}
			br.close();
			numRows = tuples.size();
			System.out.println("NumRows:  " + numRows + " NumCols:" + numCols);
		} catch (FileNotFoundException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
	}

	public ColumnMapping getColumnMapping() {
		return colMap;
	}

	public int getNumRows() {
		return numRows;
	}

	public int getNumCols() {
		return numCols;
	}

	public List<Tuple> getTuples() {
		return tuples;
	}

	public Tuple getTuple(int t) {
		return tuples.get(t);
	}

	public void removeTuple(Set<Tuple> toBeRemoved) {
		tuples.removeAll(toBeRemoved);
		numRows = tuples.size();
	}

	public void retainTuple(Set<Tuple> toBeRetained) {
		tuples.retainAll(toBeRetained);
		numRows = tuples.size();
	}

	public void insertTuples(Set<Tuple> toBeInserted) {
		tuples.addAll(toBeInserted);
		numRows = tuples.size();
	}

	/**
	 * Get cell for a row and col, both starting from 0
	 * 
	 * @param row
	 * @param col
	 * @return
	 */
	public Cell getCell(int row, int col) {
		return tuples.get(row).getCell(col);
	}

	public void setCell(int row, int col, String newValue) {
		tuples.get(row).getCell(col).setValue(newValue);
	}

	public String getTableName() {
		return inputDBPath.split("/")[inputDBPath.split("/").length - 1];
	}

	public String getSchema() {
		return schema;
	}

	public String getCurrentFolder() {
		int index = inputDBPath.lastIndexOf("/");
		StringBuilder sb = new StringBuilder(
				inputDBPath.substring(0, index + 1));
		return new String(sb);
	}

	/**
	 * Get all the constants in a particular column
	 * 
	 * @param col
	 * @return
	 */
	public Set<String> getColumnValues(int col) {
		Set<String> colValues = new HashSet<String>();
		for (int i = 0; i < numRows; i++) {
			String value = tuples.get(i).getCell(col).getValue();
			colValues.add(value);
		}
		return colValues;
	}

	/**
	 * Get all the value with duplication in a particular column
	 * 
	 * @param col
	 * @return
	 */

	public List<String> getCellsinCol(int col) {
		List<String> list = new ArrayList<String>();
		for (int i = 0; i < numRows; i++) {
			String value = tuples.get(i).getCell(col).getValue();
			list.add(value);
		}
		//System.out.println("the list is"+list);
		return list;
	}

}
