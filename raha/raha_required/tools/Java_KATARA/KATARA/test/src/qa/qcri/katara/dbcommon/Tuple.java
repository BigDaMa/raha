/**
 * Author: Xu Chu
 */
package qa.qcri.katara.dbcommon;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Tuple implements Serializable, Comparable {

	public ColumnMapping cm;
	private List<Cell> cells = new ArrayList<Cell>();

	public int tid;

	public int getTid() {
		return tid;
	}

	public Tuple(String[] values, ColumnMapping cm, int tid) {
		this.cm = cm;
		this.tid = tid;
		for (int i = 0; i < values.length; i++) {
			Cell cell = new Cell(cm.positionToType(i), values[i].trim());
			cells.add(cell);
		}
	}

	public Tuple(Tuple tuple) {
		this.cm = tuple.cm;
		for (Cell cell : tuple.cells) {
			this.cells.add(new Cell(cell));
		}
		this.tid = tuple.tid;
	}

	public List<Cell> getTuple() {
		return cells;
	}

	public int getNumCols() {
		return cells.size();
	}

	public Cell getCell(int col) {
		return cells.get(col);
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("t" + tid + ":");
		for (int i = 0; i < cells.size(); i++) {
			sb.append(cells.get(i).toString());
			if (i != cells.size() - 1)
				sb.append(",");
		}
		return new String(sb);
	}

	@Override
	public int compareTo(Object o) {
		Tuple tuple = (Tuple)o;
		if (this.tid > tuple.tid)
			return 1;
		else if (this.tid == tuple.tid)
			return 0;
		else
			return -1;
	}

}
