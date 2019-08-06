package qa.qcri.katara.common;

import java.util.Collection;
import java.util.StringTokenizer;

public class StringUtil {

	public static String polish(String s) {
		return s.replaceAll("[^\\p{Alpha}|\\s|\\-]", " ");
	}

	public static boolean isVowel(char c) {
		return c == 'a' || c == 'e' || c == 'o' || c == 'u' || c == 'i';
	}

	public static String splitCamelCase(String s) {
		String result = "";
		StringTokenizer tokenizer = new StringTokenizer(s);
		while (tokenizer.hasMoreTokens()) {
			result += " " + splitCamelCaseToken(tokenizer.nextToken());
		}
		return result.trim();
	}

	public static String splitCamelCaseToken(String s) {
		String[] words = s.split(String.format("%s|%s|%s",
				"(?<=[A-Z])(?=[A-Z][a-z])", "(?<=[^A-Z])(?=[A-Z])",
				"(?<=[A-Za-z])(?=[^A-Za-z\\-])"));
		String result = "";
		for (String word : words) {
			if (result.length() == 0) {
				result = word;
			} else {
				result += " " + word;
			}
		}
		return result;
	}
	
	public static String join(Collection list, String conjunction){
		StringBuilder sb = new StringBuilder();
		boolean first = true;
		   for (Object item : list)
		   {
		      if (first)
		         first = false;
		      else
		         sb.append(conjunction);
		      sb.append(item.toString());
		   }
		   return sb.toString();
	}
	
	public static String join(String str1, String str2, String conjunction){
		return str1+conjunction+str2;
	}
	
	public static boolean isEmptyString(String str){
		if (str == null || "".equals(str)){
			return true;
		}
		return false;
	}
	
}
