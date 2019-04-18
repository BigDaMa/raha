/**
 * Author: Xu Chu
 */
package qa.qcri.katara.dbcommon;

import java.io.Serializable;


/**
 * This class should make it easier for users to express constraints using column names
 * 
 * @author xchu
 *
 */
public class ColumnMapping implements Serializable {

	//Example: The input is A(Integer),B(String),C(Integer),D(String)
	//columnNames should [0] = A, [1] = B, [2] = C, [3] =D
	//columntypes should [0] = Integer, [1] = String, [2] = Integer, [3] = String 
	private String columnHead;
	private String[] columnNames;	//Map column names to column positions, columns positions are determined during initialization phase
	
	private String[] columnTypes;	//Map column names to column Types
	
	public ColumnMapping(String line)
	{
		init(line);
	}
	public  String getColumnHead()
	{
		return columnHead;
	}
	public  void init(String line)
	{
		columnHead = line;
		String[] columns = line.split(",");
		columnNames = new String[columns.length];
		columnTypes = new String[columns.length];
		for(int i = 0 ; i < columns.length; i++)
		{
			//System.out.println(columns[i]);
			//System.out.println(columns[i].indexOf("("));
			//System.out.println(columns[i].indexOf(")"));
			//System.out.println(columns[i].substring(0, columns[i].indexOf("(")));
			//System.out.println(columns[i].substring(columns[i].indexOf("(")+1,columns[i].indexOf(")")));
			
			if(!columns[i].contains("("))
			{
				columnNames[i] = columns[i];
				columnTypes[i] = "String";
			}
			else
			{
				columnNames[i] = columns[i];
				columnTypes[i] = "String";
				//bug here, quick fix for experiment
				
				columnNames[i] = columns[i].substring(0, columns[i].indexOf("("));
				columnTypes[i] = columns[i].substring(columns[i].indexOf("(")+1,columns[i].indexOf(")"));
			}
		}
	}
	
	
	public  int NametoPosition(String name)
	{
		for(int i =0 ; i < columnNames.length; i++)
		{
			if(columnNames[i].equalsIgnoreCase(name))
				return i;
		}
		
		System.out.println("Invalid Column Name " + name);
		return 0; //there is no such column
	}
	
	public  String NametoType(String name)
	{
		int pos = NametoPosition(name);
		
		return columnTypes[pos];
		
	}
	public String positionToName(int pos)
	{
		return columnNames[pos];
	}
	
	public  String positionToType(int pos)
	{
		return columnTypes[pos];
	}
	
	public  String[] getColumnNames()
	{
		return columnNames;
	}
}
