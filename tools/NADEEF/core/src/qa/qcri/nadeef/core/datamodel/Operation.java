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

package qa.qcri.nadeef.core.datamodel;

/**
 * Operation enumeration.
 */
public enum Operation {
    EQ(0),
    LT(1),
    GT(2),
    NEQ(3),
    LTE(4),
    GTE(5),
    CEQ(6);

    private final int value;
    private Operation(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
	
		public Operation getRevertedOpertion(){
		switch (this){
			case EQ:
				return EQ;
			case LT:
				return GT;
			case GT:
				return LT;
			case NEQ:
				return NEQ;
			case LTE:
				return GTE;
			case GTE:
				return LTE;
			case CEQ:
				return CEQ;
		}
		return null;
	}
	
	public Operation getInvertedOperation(){
		switch (this){
			case EQ:
				return NEQ;
			case LT:
				return GTE;
			case GT:
				return LTE;
			case NEQ:
				return EQ;
			case LTE:
				return GT;
			case GTE:
				return LT;
			case CEQ:
				return CEQ;
		}
		return null;
	}

	
	public String toLinquistics(){
		switch (this){
			case EQ:
				return "equals to";
			case LT:
				return "less than";
			case GT:
				return "greater than";
			case NEQ:
				return "not equal to";
			case LTE:
				return "less than or equal to";
			case GTE:
				return "greater than or equal to";
			case CEQ:
				return "not equal to";
		}
		return "unknown predicate";
	}
}
