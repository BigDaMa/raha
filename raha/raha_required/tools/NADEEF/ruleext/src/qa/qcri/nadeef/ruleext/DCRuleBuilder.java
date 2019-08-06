/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.ruleext;

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Predicate;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.tools.CommonTools;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * Template engine for DC rule.
 */
public class DCRuleBuilder extends RuleBuilder {
    private List<Predicate> predicateList;
    private String[] predicates;

    protected STGroupFile singleSTGroup;
    protected STGroupFile pairSTGroup;

    
    public List<Predicate> getPredicateList(){
    	return predicateList;
    }
    
    public DCRuleBuilder() {
        predicateList = Lists.newArrayList();
    }

    @Override
    public Collection<File> compile() throws Exception {
        File inputFile = generate().iterator().next();
        String fullPath = inputFile.getAbsolutePath();
        File classFile = new File(fullPath.replace(".java", ".class"));
        boolean alwaysCompile = NadeefConfiguration.getAlwaysCompile();
        if (alwaysCompile || !classFile.exists()) {
            CommonTools.compileFile(inputFile);
        }
        return Lists.newArrayList(classFile);
    }

    @Override
    public Collection<File> generate() throws IOException {
        singleSTGroup = new STGroupFile(
            "qa/qcri/nadeef/ruleext/template/SingleDCRuleBuilder.stg", '$', '$');
        pairSTGroup = new STGroupFile(
            "qa/qcri/nadeef/ruleext/template/PairDCRuleBuilder.stg", '$', '$');

        ST st = null;
        if (isSingle()) {
            st = singleSTGroup.getInstanceOf("dcTemplate");
        } else {
            st = pairSTGroup.getInstanceOf("dcTemplate");
        }

        if (Strings.isNullOrEmpty(ruleName)) {
            ruleName = "DefaultDC" + CommonTools.toHashCode(value.get(0));
        } else {
            // remove all the empty spaces to make it a valid class name.
            ruleName = originalRuleName.replace(" ", "");
        }

        // escape all the double quote

        st.add("DCName", ruleName);
        st.add("template", predicates);
        st.add("tableName", tableNames.get(0));
        List<File> result = Lists.newArrayList();
        File outputFile = getOutputFile();
        st.write(outputFile, null);
        result.add(outputFile);
        return result;
    }

	
	
    public boolean isSingle() {
        boolean isSingle = true;
        for (Predicate predicate : predicateList) {
            if (!predicate.isSingle()) {
                isSingle = false;
                break;
            }
        }
        return isSingle;
    }

    @Override
    protected void parse() {
        // we assume rule has only one line
        if (value == null || value.size() > 1){
            throw new IllegalArgumentException("DC must be formalized in single line");
        }

        // TODO: change to regx.
        String line = value.get(0).trim();
        if (!line.startsWith("not(")){
            throw new IllegalArgumentException("illegal header:" + line);
        }
        if (!line.endsWith(")")){
            throw new IllegalArgumentException("illegal footer:" + line);
        }
        String predicatesStr =
            line.substring(line.indexOf("not(") + 4, line.indexOf(")")).trim();
        predicates = predicatesStr.split("&");
        String tableName = tableNames.get(0);
        for (int i = 0; i < predicates.length; i++) {
			predicateList.add(Predicate.valueOf(predicates[i].trim(), tableName));
        }
    }
}