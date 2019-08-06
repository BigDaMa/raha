package qa.qcri.katara.kbcommon.pattern.simple;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.*;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class TableSemantics {

	public static final String REL_REVERSED_TAG = "REVERSED_";
	
	public Map<String,String> col2Type = new HashMap<String,String>();

	public Map<String,String> col2Rel = new HashMap<String,String>();
	
	public double score;
	public Map<String,Double> col2TypeScore = new HashMap<String,Double>();
	public Map<String,Double> col2RelScore = new HashMap<String,Double>();
	public Map<String,Double> col2RelDomainCoherenceScore = new HashMap<String,Double>();
	public Map<String,Double> col2RelRangeCoherenceScore = new HashMap<String,Double>();
	
	/*
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("Score: " + score + "\n ");
		for(String col: col2Type.keySet())
		{
			sb.append("Column " + col + " has Type: " + col2Type.get(col)
					+ "\n ");
		}
		for(String col: col2Rel.keySet())
		{
			sb.append("Column pair " + col + " has Rel: " + col2Rel.get(col) 
					 + "\n");
		}
		return new String(sb);
	}
	*/
	
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("Total Score: " + score + "\n ");
		for(String col: col2Type.keySet())
		{
			sb.append("Column " + col + " has Type: " + col2Type.get(col)
					+ "	Type Score: " + col2TypeScore.get(col) + "\n ");
		}
		for(String col: col2Rel.keySet())
		{
			sb.append("Column pair " + col + " has Rel: " + col2Rel.get(col) 
					+ "	Rel Score: "  + col2RelScore.get(col) 
					+ "	Rel Domain Coherence Score: "  + col2RelDomainCoherenceScore.get(col) 
					+ "	Rel Range Coherence Score: "  + col2RelRangeCoherenceScore.get(col)  + "\n");
		}
		return new String(sb);
	}
	
	
	
	public static void serialize(TableSemantics ts, String fileName)
	{
		Writer writer = null;
		try {
			writer = new FileWriter(fileName);
		    Gson gson = new GsonBuilder().create();
	        gson.toJson(ts, writer);
	        writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

       
	}
	
	public static TableSemantics deSerialize(String fileName)
	{
		
		TableSemantics ts = null;
		
		try(Reader reader = new FileReader(fileName))
		{
            Gson gson = new GsonBuilder().create();
            ts = gson.fromJson(reader, TableSemantics.class);
        } catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ts;
	}
}
