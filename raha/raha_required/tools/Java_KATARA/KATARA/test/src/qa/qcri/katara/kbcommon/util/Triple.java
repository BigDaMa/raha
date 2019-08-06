package qa.qcri.katara.kbcommon.util;

/**
 * 
 * @author Yin Ye
 *
 * @param <U> clazz of first cell
 * @param <V> clazz of second cell
 * @param <W> clazz of third cell
 * 
 */
public class Triple<U, V, W> {
	
	public Triple(U u, V v, W w){
		this.u = u;
		this.v = v;
		this.w = w;
	}
	
	private U u;
	private V v;
	private W w;

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((u == null) ? 0 : u.hashCode());
		result = prime * result + ((v == null) ? 0 : v.hashCode());
		result = prime * result + ((w == null) ? 0 : w.hashCode());
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
		Triple other = (Triple) obj;
		if (u == null) {
			if (other.u != null)
				return false;
		} else if (!u.equals(other.u))
			return false;
		if (v == null) {
			if (other.v != null)
				return false;
		} else if (!v.equals(other.v))
			return false;
		if (w == null) {
			if (other.w != null)
				return false;
		} else if (!w.equals(other.w))
			return false;
		return true;
	}

	public U getU() {
		return u;
	}

	public void setU(U u) {
		this.u = u;
	}

	public V getV() {
		return v;
	}

	public void setV(V v) {
		this.v = v;
	}

	public W getW() {
		return w;
	}

	public void setW(W w) {
		this.w = w;
	}

}
