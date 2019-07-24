package qa.qcri.katara.kbcommon.pattern.simple;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.ArrayList;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;



public class PatternSerializationHolistic {
	public String tableName;
	public ArrayList<TableSemantics> tss = new ArrayList<TableSemantics>();
	
	
	public PatternSerializationHolistic(String tableName,ArrayList<TableSemantics> tss)
	{
		this.tableName = tableName;
		this.tss = tss;
	}
	
	public static void serialize(PatternSerializationHolistic ps, String fileName)
	{
		Writer writer = null;
		try {
			writer = new FileWriter(fileName);
		    Gson gson = new GsonBuilder().serializeSpecialFloatingPointValues().create(); //create();
	        gson.toJson(ps, writer);
	        writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

       
	}
	
	public static PatternSerializationHolistic deSerialize(String fileName)
	{
		PatternSerializationHolistic ps = null;
		
		try(Reader reader = new FileReader(fileName))
		{
            Gson gson = new GsonBuilder().create();
            ps = gson.fromJson(reader, PatternSerializationHolistic.class);
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
		for(int i = 0 ; i < tss.size(); i++)
		{
			sb.append("*********************************************************#" + (i+1) + "\n");
			sb.append(tss.get(i).toString());
		}
		return new String(sb);
		
	}
}
