/**
 * Author: Xu Chu
 * Pattern Serialization / DeSerialization
 */
package qa.qcri.katara.kbcommon.pattern.simple;

import java.util.*;
import java.io.*;

import com.google.gson.*;

public class PatternSerialization {

	//the table this pattern is about
	public String tableName;
	
	//Example: 0->Country->0.8, 0->Economy->0.2
	public Map<String,Map<String,Double>> col2Type2Score = new HashMap<String,Map<String,Double>>();
	
	//Example: 0,1->hasCapital->0.7, 1,0->isLocatedIn->0.3
	public Map<String,Map<String,Double>> col2Rel2Score = new HashMap<String,Map<String,Double>>();
	
	
	public PatternSerialization(String tableName,Map<String,Map<String,Double>> col2Type2Score,
			Map<String,Map<String,Double>> col2Rel2Score)
	{
		this.tableName = tableName;
		this.col2Type2Score = col2Type2Score;
		this.col2Rel2Score = col2Rel2Score;
	}
	
	public static void serialize(PatternSerialization ps, String fileName)
	{
		Writer writer = null;
		try {
			writer = new FileWriter(fileName);
		    Gson gson = new GsonBuilder().create();
	        gson.toJson(ps, writer);
	        writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

       
	}
	
	public static PatternSerialization deSerialize(String fileName)
	{
		PatternSerialization ps = null;
		
		try(Reader reader = new FileReader(fileName))
		{
            Gson gson = new GsonBuilder().create();
            ps = gson.fromJson(reader, PatternSerialization.class);
        } catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ps;
	}
	
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("tableName: " + tableName);
		sb.append("\n");
		
		for(String col: col2Type2Score.keySet())
		{
			sb.append("Column " + col + " has types and scores: \n");
			for(String type: col2Type2Score.get(col).keySet())
			{
				sb.append( type + " : " + col2Type2Score.get(col).get(type));
				sb.append("\n");
			}
			sb.append("\n");
		}
		
		for(String col: col2Rel2Score.keySet())
		{
			sb.append("Rel " + col + " has rels and scores: \n");
			for(String rel: col2Rel2Score.get(col).keySet())
			{
				sb.append(rel + " : " + col2Rel2Score.get(col).get(rel));
				sb.append("\n");
			}
			sb.append("\n");
		}
		
		return new String(sb);
	}
}
