package qa.qcri.katara.kbcommon.pattern.simple;

public class TypeRelCoherence {

	private String type;
	private String rel;
	private double coherence;
	private boolean isDomainType;
	String delimi = ",";
	
	public TypeRelCoherence(String type, String rel, boolean isDomainType, double coherence)
	{
		this.type = type;
		this.rel = rel;
		this.isDomainType = isDomainType;
		this.coherence = coherence;
	}
	public TypeRelCoherence(String line)
	{
		String[] temp = line.split(delimi);
		if(temp.length == 4){
			this.type = temp[0];
			this.rel = temp[1];
			this.isDomainType = Boolean.valueOf(temp[2]);
			this.coherence = Double.valueOf(temp[3]);
		}else{
			this.coherence = Double.valueOf(temp[temp.length - 1]);
			this.isDomainType = Boolean.valueOf(temp[temp.length - 2]);
			StringBuilder sb1 = new StringBuilder();
			sb1.append(temp[0]);
			
			StringBuilder sb2 = new StringBuilder();
			int i = 1;
			for(i = 1; i < temp.length - 2; i++){
				if(!temp[i].startsWith("http")){
					sb1.append(delimi + temp[i]);
				}else{
					break;
				}
			}
			for(; i < temp.length - 2; i++){
				if(temp[i].startsWith("http")){
					sb2.append(temp[i]);
				}else{
					sb2.append(delimi + temp[i]);
				}
			}
			this.type = new String(sb1);
			this.rel = new String(sb2);
		}
		
		
	}
	public String getType()
	{
		return type;
	}
	
	
	public String getRel() {
		return rel;
	}
	public void setRel(String rel) {
		this.rel = rel;
	}
	public double getCoherence() {
		return coherence;
	}
	public void setCoherence(double coherence) {
		this.coherence = coherence;
	}
	public boolean isDomainType() {
		return isDomainType;
	}
	public void setDomainType(boolean isDomainType) {
		this.isDomainType = isDomainType;
	}
	public String getDelimi() {
		return delimi;
	}
	public void setDelimi(String delimi) {
		this.delimi = delimi;
	}
	public void setType(String type) {
		this.type = type;
	}
	public String toString()
	{
		
		return type + delimi + rel + delimi + isDomainType + delimi + coherence;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((delimi == null) ? 0 : delimi.hashCode());
		result = prime * result + (isDomainType ? 1231 : 1237);
		result = prime * result + ((rel == null) ? 0 : rel.hashCode());
		result = prime * result + ((type == null) ? 0 : type.hashCode());
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
		TypeRelCoherence other = (TypeRelCoherence) obj;
		if (delimi == null) {
			if (other.delimi != null)
				return false;
		} else if (!delimi.equals(other.delimi))
			return false;
		if (isDomainType != other.isDomainType)
			return false;
		if (rel == null) {
			if (other.rel != null)
				return false;
		} else if (!rel.equals(other.rel))
			return false;
		if (type == null) {
			if (other.type != null)
				return false;
		} else if (!type.equals(other.type))
			return false;
		return true;
	}
	
	

}
