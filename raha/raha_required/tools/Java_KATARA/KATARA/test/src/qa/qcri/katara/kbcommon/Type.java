package qa.qcri.katara.kbcommon;

import com.google.gson.JsonObject;

/**
 * 
 * Define the type information for each column in table
 * @author Yin Ye
 *
 */
public class Type {
	
	private String columnName;
	private String type;

	public Type(String columnName, String type) {
		this.columnName = columnName;
		this.type = type;
	}

	public String getColumnName() {
		return columnName;
	}

	public void setColumnName(String columnName) {
		this.columnName = columnName;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((columnName == null) ? 0 : columnName.hashCode());
		result = prime * result + ((type == null) ? 0 : type.hashCode());
		return result;
	}

	public JsonObject toJson(){
		JsonObject obj = new JsonObject();
		obj.addProperty("type", getType());
		obj.addProperty("column", getColumnName());
		return obj;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Type other = (Type) obj;
		if (columnName == null) {
			if (other.columnName != null)
				return false;
		} else if (!columnName.equals(other.columnName))
			return false;
		if (type == null) {
			if (other.type != null)
				return false;
		} else if (!type.equals(other.type))
			return false;
		return true;
	}

}
