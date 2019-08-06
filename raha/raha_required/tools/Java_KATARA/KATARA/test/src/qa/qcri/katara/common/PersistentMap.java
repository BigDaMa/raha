package qa.qcri.katara.common;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import qa.qcri.crowdservice.common.Jsonizable;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * 
 * @author yye
 * 
 * @param <K>
 * @param <V>
 */
public class PersistentMap<K extends Jsonizable, V extends Jsonizable>
		implements Map<K, V>, AutoCloseable {

	private static Object lock = new Object();

	class Entry implements Jsonizable<Entry> {
		private JsonObject key;
		private JsonObject value;

		public JsonObject getKey() {
			return key;
		}

		public void setKey(JsonObject key) {
			this.key = key;
		}

		public JsonObject getValue() {
			return value;
		}

		public void setValue(JsonObject value) {
			this.value = value;
		}

		public Entry() {

		}

		public Entry(JsonObject key, JsonObject value) {
			this.key = key;
			this.value = value;
		}

		@Override
		public Entry fromJson(JsonObject object) {
			Entry entry = new Entry();
			JsonElement objKey = object.get("key");
			JsonElement objValue = object.get("value");
			entry.setKey(objKey.getAsJsonObject());
			entry.setValue(objValue.getAsJsonObject());
			return entry;
		}

		@Override
		public JsonObject toJson() {
			JsonObject jsonObject = new JsonObject();
			jsonObject.add("key", key);
			jsonObject.add("value", value);
			return jsonObject;
		}
	}

	private Map<K, V> map = new HashMap<K, V>();

	private String filePath;

	@SuppressWarnings("unchecked")
	public PersistentMap(String filePath1, K k, V v) throws IOException {
		this.filePath = filePath1;
		// save relationships
		File file = new File(filePath);
		if (!file.exists()) {
		} else {
			BufferedReader reader = null;
			try {
				reader = new BufferedReader(new FileReader(filePath));
				String str = null;
				JsonParser parser = new JsonParser();
				Entry entryFactory = new Entry();
				while ((str = reader.readLine()) != null) {
					JsonObject o = (JsonObject) parser.parse(str);
					Entry entry = entryFactory.fromJson(o);
					K keyInstance = (K) k.fromJson(entry.getKey());
					V valueInstance = (V) v.fromJson(entry.getValue());
					map.put(keyInstance, valueInstance);
				}
			} finally {
				reader.close();
			}
		}
	}

	// TODO, we should have better write back strategy, lock strategy is not good as well
	public void commit() throws IOException {
		synchronized (lock) {
			FileUtil.refreshFile(filePath);
			BufferedWriter outputStream = null;
			try {
				outputStream = new BufferedWriter(new FileWriter(filePath));
				for (K k : map.keySet()) {
					Entry entry = new Entry(k.toJson(), map.get(k).toJson());
					outputStream.write(entry.toJson().toString());
					outputStream.newLine();
				}
			} finally {
				outputStream.close();
			}
		}
	}

	@Override
	public int size() {
		return map.size();
	}

	@Override
	public boolean isEmpty() {
		return map.isEmpty();
	}

	@Override
	public boolean containsKey(Object key) {
		return map.containsKey(key);
	}

	@Override
	public boolean containsValue(Object value) {
		return map.containsValue(value);
	}

	@Override
	public V get(Object key) {
		return map.get(key);
	}

	@Override
	public V put(K key, V value) {

		return map.put(key, value);
	}

	@Override
	public V remove(Object key) {
		return map.remove(key);
	}

	@Override
	public void clear() {
		map.clear();

	}

	@Override
	public Set<K> keySet() {
		return map.keySet();
	}

	@Override
	public Collection<V> values() {
		return map.values();
	}

	@Override
	public Set<java.util.Map.Entry<K, V>> entrySet() {
		return map.entrySet();
	}

	@Override
	public void putAll(Map<? extends K, ? extends V> m) {
		map.putAll(m);
	}

	@Override
	public void close() throws Exception {
		commit();
	}
}