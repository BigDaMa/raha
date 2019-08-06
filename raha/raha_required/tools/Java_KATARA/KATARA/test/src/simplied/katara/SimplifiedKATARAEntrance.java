package simplied.katara;
import java.io.*;
import java.util.*;
import qa.qcri.katara.dbcommon.Table;
import qa.qcri.katara.kbcommon.KBReader;
import qa.qcri.katara.kbcommon.KnowledgeDatabaseConfig;
import qa.qcri.katara.kbcommon.PatternDiscoveryException;
import java.nio.file.Path;
import java.nio.file.Paths;


public class SimplifiedKATARAEntrance {

	public static void main(String[] args) throws Exception {


		//setup

		Scanner scanner = new Scanner(System.in);
		String rdb = scanner.next();

		Path path=Paths.get("");
		String myKbPath[]=path.toAbsolutePath().toString().split("/");
		// Create temporary mykb
		String kb="/tmp/mykb";
		File directory = new File(String.valueOf(kb));
		if(!directory.exists()) {
			directory.mkdir();
		}
		//for (int i=0;i<myKbPath.length-1;i++){
		//	kb+=myKbPath[i]+"/";
		//}
		//kb+="abstraction-layer/tools/KATARA/mykb";//abstraction-layer/tools/

		String output_errors_file = "katara_output-" + rdb;//rdb+


		String domainSpecificKB = scanner.next();
		//System.out.println(kb);

		//run KATARA
		KnowledgeDatabaseConfig.setDataDirectoryBase(kb);
		KnowledgeDatabaseConfig.KBStatsDirectoryBase = kb + "Stats";
		KnowledgeDatabaseConfig.frequentPercentage = 0.5;

		//Let's only deal with the first 1000 rows
		Table table = new Table(rdb,Integer.MAX_VALUE);

		KBReader reader = new KBReader();
		SimplifiedPatternDiscovery spd = new SimplifiedPatternDiscovery(table,reader,domainSpecificKB);
		spd.errorDetection(true);
		spd.print_errors(output_errors_file);
		reader.close();
	}
}
