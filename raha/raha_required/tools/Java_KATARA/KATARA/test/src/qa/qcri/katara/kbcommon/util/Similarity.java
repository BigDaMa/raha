/**
 * Author: Xu Chu
 */
package qa.qcri.katara.kbcommon.util;

/**
 * Check whether two strings are similar with each other
 * @author xchu
 *
 */
public class Similarity {

	
	private static int threadshold = 3;
	public static boolean editDistance(String s1, String s2)
	{
		
		return false;
	}
	
	
	//This is where you set the threadshuld
	public static boolean levenshteinDistance(String s1, String s2)
	{
		int dis = computeLevenshteinDistance(s1,s2);
		if(dis <= threadshold)
			return true;
		else
			return false;
	}
	
	public static boolean cosineDistance(String s1, String s2)
	{
		throw new UnsupportedOperationException();
	}
	
	
	
	private static int minimum(int a, int b, int c) 
	{
        return Math.min(Math.min(a, b), c);
	}

	public static int computeLevenshteinDistance(CharSequence str1,
                CharSequence str2) {
        int[][] distance = new int[str1.length() + 1][str2.length() + 1];

        for (int i = 0; i <= str1.length(); i++)
                distance[i][0] = i;
        for (int j = 0; j <= str2.length(); j++)
                distance[0][j] = j;

        for (int i = 1; i <= str1.length(); i++)
                for (int j = 1; j <= str2.length(); j++)
                        distance[i][j] = minimum(
                                        distance[i - 1][j] + 1,
                                        distance[i][j - 1] + 1,
                                        distance[i - 1][j - 1]
                                                        + ((str1.charAt(i - 1) == str2.charAt(j - 1)) ? 0
                                                                        : 1));

        return distance[str1.length()][str2.length()];
	}
	
}
