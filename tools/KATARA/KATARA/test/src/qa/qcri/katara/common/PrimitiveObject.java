package qa.qcri.katara.common;

import qa.qcri.crowdservice.common.Jsonizable;

import com.google.gson.JsonObject;

public class PrimitiveObject implements Jsonizable<PrimitiveObject> {

	private String str;
	
	@Override
	public PrimitiveObject fromJson(JsonObject object) {
		PrimitiveObject obj = new PrimitiveObject(object.get("val").getAsString());
		return obj;
	}

	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((str == null) ? 0 : str.hashCode());
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		PrimitiveObject other = (PrimitiveObject) obj;
		if (str == null) {
			if (other.str != null)
				return false;
		} else if (!str.equals(other.str))
			return false;
		return true;
	}


	public String getStr() {
		return str;
	}


	public void setStr(String str) {
		this.str = str;
	}


	@Override
	public JsonObject toJson() {
		JsonObject jsonObject = new JsonObject();
		jsonObject.addProperty("val", str);
		return jsonObject;
	}

	public PrimitiveObject() {
		// used as factory
	}
	
	public PrimitiveObject(String str) {
		this.str = str;
	}
}
