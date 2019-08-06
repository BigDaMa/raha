package simplied.katara;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.supercsv.io.CsvBeanWriter;
import org.supercsv.io.ICsvBeanWriter;
import org.supercsv.prefs.CsvPreference;
import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.dbcommon.Tuple;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.pattern.simple.TableSemantics;
import org.apache.commons.lang3.StringUtils;


public class SimplifiedPatternDiscovery {

	Table table;
	KBReader reader;
	static String repairvalue="";

	Scanner scan =new Scanner(System.in);

	//This is the result
	Map<String,String> col2TypeRel = new HashMap<String,String>();
	Map<String,Set<Tuple>> col2SupportingTuples = new HashMap<String,Set<Tuple>>();

	public SimplifiedPatternDiscovery(Table table, KBReader reader, String domainSpecifciKBDir){
		this.table = table;
		this.reader = reader;



		if(domainSpecifciKBDir != null){
			try {
				usingDomainSpecificKB(domainSpecifciKBDir,true);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//System.out.println("if there is no kb is still work");
		if(true){
			return;
		}


		PatternGenerationTFIDF tfidf = new PatternGenerationTFIDF(reader,table);


		//get the type of columns
		System.out.println("Get the candidate type of columns..." );


		for (int col = 0; col < table.getNumCols(); col++) {

			//if covered, skip


			if(domainSpecificCoveredColumns.contains(col))
				continue;

			Map<String, Set<Tuple>> result = tfidf.getCandidateTypes(col);

			final String colFinal = String.valueOf(col);


			final Map<String,Double> type2Score = tfidf.setCandidateTypeScore(colFinal, result);

			ArrayList<String> typesRanked = new ArrayList<String>(
					result.keySet());
			// Limit the size of each list
			ArrayList<String> typesRankedLimit = new ArrayList<String>();
			for (int i = 0; i < typesRanked.size(); i++) {
				String type = typesRanked.get(i);

				double sup = (double) result.get(type).size()
						/ table.getNumRows();
				if (sup >= KnowledgeDatabaseConfig.frequentPercentage) {
					typesRankedLimit.add(type);
				}

				if(typesRankedLimit.size() > 2)
					break;
			}
			if (typesRankedLimit.isEmpty()) {
				continue; // If the candidate set is empty, do not add it to
				// ranked list
			}

			Collections.sort(typesRankedLimit, new Comparator<String>() {
				public int compare(String arg0, String arg1) {
					if (type2Score.get(arg0) > type2Score.get(arg1))
						return -1;
					else if (type2Score.get(arg0) < type2Score.get(arg1))
						return 1;
					/*else
						return 0;*/
					else if (RankJoinKBStats.moreSpecificType(arg0, arg1)) {
						return -1;
					} else
						return 1;
				}

			});
			String type = user_verification_column_type(col,typesRankedLimit);

			if(type != null){
				col2TypeRel.put(colFinal, type);
				col2SupportingTuples.put(colFinal, result.get(type));
			}

		}

		//get relationship between column pairs
		System.out.println("Get the relationship between column pairs..." );
		if(true){
			return;
		}

		for (int col1 = 0; col1 < table.getNumCols(); col1++) {
			for (int col2 = col1 + 1; col2 < table.getNumCols(); col2++) {
				// get the candidate rels, and the supporting type


				if(domainSpecificCoveredColumns.contains(col1) && domainSpecificCoveredColumns.contains(col2))
					continue;


				Map<String, Set<Tuple>> result = tfidf.getCandidateRels(col1, col2);

				final String colFinal = col1 + "," + col2;
				if (result.isEmpty()) {
					continue; // If the candidate rel for a column pair is
					// empty, do not add it
				}



				// set the score of each candidate rels
				final Map<String,Double> rel2Score =  tfidf.setCandidateRelScore(colFinal, result);


				// rank the candidat rel based on the score

				ArrayList<String> relsRanked = new ArrayList<String>(
						result.keySet());
				// limit the size of each rel
				ArrayList<String> relsRankedLimit = new ArrayList<String>();
				for (int i = 0; i < relsRanked.size(); i++) {
					String rel = relsRanked.get(i);
					double sup = (double) result
							.get(rel).size()
							/ table.getNumRows();
					// double score = col2Rel2Score.get(col1 + "," +
					// col2).get(rel);
					if (sup >= KnowledgeDatabaseConfig.frequentPercentage) {
						relsRankedLimit.add(rel);
					}
				}

				if (relsRankedLimit.isEmpty())
					continue;

				System.out.println("Columns: " + col1 + " , " + col2
						+ " has # candidate rels: above minimum fre "
						+ KnowledgeDatabaseConfig.frequentPercentage + " is :"
						+ relsRankedLimit.size());

				Collections.sort(relsRankedLimit, new Comparator<String>() {
					public int compare(String arg0, String arg1) {
						if (rel2Score.get(arg0) > rel2Score.get(arg1))
							return -1;
						else if (rel2Score.get(arg0) < rel2Score.get(arg1))
							return 1;
						else if (RankJoinKBStats.moreSpecificRel(arg0, arg1)) {
							return -1;
						} else
							return 1;
					}

				});


				String rel = user_verification_column_pair_rel(colFinal,relsRankedLimit);
				if(rel != null){
					col2TypeRel.put(colFinal, rel);
					col2SupportingTuples.put(colFinal, result.get(rel));
				}

			}
		}
	}

	public String user_verification_column_type(int col, ArrayList<String> typesRankedLimit){



		System.out.println("Start User Verification for the Type of Column:  " + col);
		System.out.println("The types are ranked in decreasing order of likelihood");
		System.out.println("Enter Y to accept this type, N to examine next type, T to terminate");
		for(int i = 0; i < typesRankedLimit.size(); i++){
			String type = typesRankedLimit.get(i);
			System.out.println("\t Is it (Y/N/T) :  " + type);
			String answer = scan.nextLine();
			if(answer.equalsIgnoreCase("Y")){

				return type;
			}else if(answer.equalsIgnoreCase("N")){
				continue;
			}else if(answer.equalsIgnoreCase("T")){

				return null;
			}else{
				continue;
			}
		}
		return null;
	}
	public String user_verification_column_pair_rel(String colFinal, ArrayList<String> relsRankedLimit){


		System.out.println("Start User Verification for the relationship between column pair:  " + colFinal);
		System.out.println("The relationships are ranked in decreasing order of likelihood");
		System.out.println("Enter Y to accept this, N to examine next, T to terminate");
		for(int i = 0; i < relsRankedLimit.size(); i++){
			String rel = relsRankedLimit.get(i);
			System.out.println("\t Is it (Y/N/T) :  " + rel);
			String answer = scan.nextLine();
			if(answer.equalsIgnoreCase("Y")){

				return rel;
			}else if(answer.equalsIgnoreCase("N")){
				continue;
			}else if(answer.equalsIgnoreCase("T")){

				return null;
			}else{
				continue;
			}
		}
		return null;
	}


	Map<String,Set<Tuple>> col2Errors = new HashMap<String,Set<Tuple>>();
	Map<String,String> col2Errorsrepair = new HashMap<String,String>();
	/**
	 * Do we allow user input for error detection
	 * @param userInput
	 */

	public void errorDetection(boolean userInput	){



		for(String col: col2TypeRel.keySet()){

			Set<Tuple> supportingTuples = col2SupportingTuples.get(col);
			col2Errors.put(col, new HashSet<Tuple>());

			if(!col.contains(",")){
				String type = col2TypeRel.get(col);
				Set<String> supportingValues = new HashSet<String>();

				for(Tuple t: table.getTuples()){

					if(supportingTuples.contains(t)){

						continue;
					}else{

						if(userInput){

							String value = t.getCell(Integer.valueOf(col)).getValue();
							if(supportingValues.contains(value))
								continue;
							System.out.println("Is [ " +  value + " ] of type [ " + type);
							String answer = scan.nextLine();
							if(answer.equalsIgnoreCase("Y")){
								supportingValues.add(value);
							}else if(answer.equalsIgnoreCase("N")){
								col2Errors.get(col).add(t);
							}
						}else{
							col2Errors.get(col).add(t);
						}

					}
				}
			}else{
				String rel = col2TypeRel.get(col);
				Set<String> supportingValues = new HashSet<String>();
				int col1 = Integer.valueOf(col.split(",")[0]);
				int col2 = Integer.valueOf(col.split(",")[1]);
				if(rel.startsWith(TableSemantics.REL_REVERSED_TAG)){
					col1 = Integer.valueOf(col.split(",")[1]);
					col2 = Integer.valueOf(col.split(",")[0]);
					rel = rel.substring(TableSemantics.REL_REVERSED_TAG.length());
				}

				for(Tuple t: table.getTuples()){
					if(supportingTuples.contains(t)){
						continue;
					}else{
						if(userInput){
							String value1 = t.getCell(Integer.valueOf(col1)).getValue();
							String value2 = t.getCell(Integer.valueOf(col2)).getValue();
							if(supportingValues.contains(value1 + "," + value2))
								continue;

							System.out.println("Is it true that [ " + value1 + "] " + rel + " [ " + value2 + "]" );
							String answer = scan.nextLine();
							if(answer.equalsIgnoreCase("Y")){
								supportingValues.add(value1 + "," + value2);
							}else if(answer.equalsIgnoreCase("N")){
								col2Errors.get(col).add(t);
							}
						}else{
							col2Errors.get(col).add(t);
						}

					}
				}
			}

		}
	}



	public void print_errors(String error_file) throws IOException{

		/// ------------------------------------
		ICsvBeanWriter supercsvWriter = null;
//		//PrintWriter out = new PrintWriter(new FileWriter(error_file));
//		CSVWriter out = new CSVWriter(new FileWriter(error_file),CSVWriter.DEFAULT_SEPARATOR, CSVWriter.NO_QUOTE_CHARACTER);

		try {
			supercsvWriter = new CsvBeanWriter(new FileWriter(error_file), CsvPreference.STANDARD_PREFERENCE);
			final String[] nameMapping = new String[] { "row", "column", "value" };

			// write the beans
			for(String col: col2Errors.keySet()){
				for(Tuple t: col2Errors.get(col)){
					if(!col.contains(",")){
						int col1 = Integer.valueOf(col);
						//out.println( (t.getTid() +1 ) + "," + (col1 )+ ","+ col2Errorsrepair.get(t.getTid()+","+col1));//+1

						supercsvWriter.write(new PrintableError(t.getTid()+1, col1, col2Errorsrepair.get(t.getTid()+","+col1)), nameMapping);
//					out.writeNext((t.getTid() +1 ) +","+ (col1 )+","+ col2Errorsrepair.get(t.getTid()+","+col1)));

						//out.println("The following cell is wrong: Cell[" + t.getTid() + "]["+col+"]=" + t.getCell(Integer.valueOf(col)).getValue());
					}else{
						int col1 = Integer.valueOf(col.split(",")[0]);
						int col2 = Integer.valueOf(col.split(",")[1]);

						//out.println( (t.getTid() +1) + "," + (col1)+ ","+ col2Errorsrepair.get(t.getTid()+","+col1));//+1
						//out.println( (t.getTid()+1) + "," + (col2)+ ","+ col2Errorsrepair.get(t.getTid()+","+col2));//+1

						supercsvWriter.write(new PrintableError(t.getTid() +1, col1, col2Errorsrepair.get(t.getTid()+","+col1)), nameMapping);
						supercsvWriter.write(new PrintableError(t.getTid() +1, col2, col2Errorsrepair.get(t.getTid()+","+col2)), nameMapping);
//					out.writeNext((t.getTid() +1) +","+ (col1)+","+ col2Errorsrepair.get(t.getTid()+","+col1));
//					out.writeNext((t.getTid()+1) +","+ (col2)+","+ col2Errorsrepair.get(t.getTid()+","+col2));

//					out.println("There is at least one Error in the following two cells: Cell[" + t.getTid() + "]["+col1+"]=" + t.getCell(col1).getValue()
//							+ "\t Cell[" + t.getTid() + "]["+col2+"]=" + t.getCell(col2).getValue());

					}
				}
			}
		}

		finally {
			if( supercsvWriter != null ) {
				supercsvWriter.close();
			}
		}

		
		File file = new File(error_file);
		if(file.length() == 0) {
			file.delete();
			System.out.println("file has been deleted");
		}

	}


	/**
	 * The following is to use domain specific knowledge bases
	 */

	List<HashSet<String>> domainSpecificTypes = new ArrayList<HashSet<String>>();

	Map<String,Map<String,String>> rel2sub2obj = new HashMap<String,Map<String,String>>();

	HashSet<Integer> domainSpecificCoveredColumns = new HashSet<Integer>();

	public void usingDomainSpecificKB(String kbDir, boolean ignoreCase) throws IOException{
		loadDomainSpecificKB(kbDir);


		for(int i = 0; i < table.getNumCols(); i++){
			usingDomainSpecificKB_col_type(i,ignoreCase);
		}
		for(int i = 0; i < table.getNumCols(); i++)
			for(int j = 0; j < table.getNumCols(); j++){
				if(i == j)
					continue;

				usingDomainSpecificKB_colpair_rel(i,j,ignoreCase);
			}
	}
	private void usingDomainSpecificKB_colpair_rel(int i, int j, boolean ignoreCase){
		for(String rel: rel2sub2obj.keySet()){
			int count = 0;
			for(Tuple t: table.getTuples()){
				String coli = t.getCell(i).getValue();
				String colj = t.getCell(j).getValue();

				if(rel2sub2obj.get(rel).containsKey(coli) && rel2sub2obj.get(rel).get(coli).equals(colj)){
					count++;
				}
			}
			double coverage = (double) count / table.getNumRows();

			if(coverage >= 0.15){
				HashSet<Tuple> errorTuples = new HashSet<Tuple>();
				HashSet<String> errorTuplesrepair = new HashSet<String>();
				//find the right domain specific type

				for(Tuple t: table.getTuples()){
					String coli = t.getCell(i).getValue();
					String colj = t.getCell(j).getValue();
					if(rel2sub2obj.get(rel).containsKey(coli) && rel2sub2obj.get(rel).get(coli).equals(colj)){

					}else{
						errorTuples.add(t);


						// we triying to suggest the repair

						if(rel2sub2obj.get(rel).containsKey(coli)) {
							repairvalue=rel2sub2obj.get(rel).get(coli);
							col2Errorsrepair.put(t.getTid() + "," + j,repairvalue);

						}

					}
				}
				col2Errors.put(i + "," + j, errorTuples);

			}
		}

	}


	public double compareStrings(String stringA, String stringB) {
		return StringUtils.getJaroWinklerDistance(stringA, stringB);
	}



	private void usingDomainSpecificKB_col_type(int i, boolean ignoreCase){
		List<String> values = table.getCellsinCol(i);

		//check the domain KBs
		int k = 0;
		for(k = 0; k < domainSpecificTypes.size(); k++){

			HashSet<String> domainValues = domainSpecificTypes.get(k);
			HashSet<String> lowercaseDomainValues = new HashSet<String>();

			for(String value: domainValues){
				lowercaseDomainValues.add(value.toLowerCase());

			}

			int count = 0;


			for(String value: values){
				if(ignoreCase && lowercaseDomainValues.contains(value.toLowerCase())){

					count++;

				}else if(domainValues.contains(value)) {

					count++;
				}
			}
			double coverage = (double) count / values.size();

			if(coverage > 0.2){
				HashSet<Tuple> errorTuples = new HashSet<Tuple>();
				//find the right domain specific type
				for(Tuple t: table.getTuples()){
					String value = t.getCell(Integer.valueOf(i)).getValue();

					if(ignoreCase && lowercaseDomainValues.contains(value.toLowerCase())){

					}else if(domainValues.contains(value)) {

					}else{
						errorTuples.add(t);

//						//##################
//						// we trying to report the repair
//						String a = value.toLowerCase();
//						double mindistance=0;
//						String index="";
//						int number=0;
//						for (String potentialrepair:domainValues){
//							String b = potentialrepair.toLowerCase();
//							double ll=compareStrings(a,b);
//							if (ll>=mindistance){
//								mindistance=ll;
//								index=potentialrepair;
//							}
//						}

						String index="";
						// index is the value that we suggest as error
						col2Errorsrepair.put(t.getTid() + "," + i,index);
					}
				}
				col2Errors.put(String.valueOf(i), errorTuples);

				domainSpecificCoveredColumns.add(i);


				break;
			}

		}
	}


	private void loadDomainSpecificKB(String kbDir) throws IOException{
		File folder = new File(kbDir);
		File[] listOfFiles = folder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile() && listOfFiles[i].getAbsolutePath().endsWith(".type.txt")) {
				HashSet<String> oneType = new HashSet<String>();
				BufferedReader reader = new BufferedReader(new FileReader(listOfFiles[i]));
				String line = null;
				while((line = reader.readLine())!=null){
					oneType.add(line);
				}
				reader.close();
				domainSpecificTypes.add(oneType);
			} else if (listOfFiles[i].isFile() && listOfFiles[i].getAbsolutePath().endsWith(".rel.txt")) {
				HashSet<String> oneType = new HashSet<String>();

				BufferedReader reader = new BufferedReader(new FileReader(listOfFiles[i]));
				String line = null;
				while((line = reader.readLine())!=null){
					String[] splits = line.split("\t");
					if(splits.length < 3)
						continue;
					String s = splits[0];
					String p = splits[1];
					String o = splits[2];
					oneType.add(s);
					if(rel2sub2obj.containsKey(p)){
						rel2sub2obj.get(p).put(s, o);
					}else{
						HashMap<String,String> temp = new HashMap<String,String>();
						temp.put(s, o);
						rel2sub2obj.put(p, temp);
					}
				}
				reader.close();

				domainSpecificTypes.add(oneType);
			}
			else if (listOfFiles[i].isDirectory()) {
				System.out.println("Directory " + listOfFiles[i].getName());
			}
		}
	}


}
