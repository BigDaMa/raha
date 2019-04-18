package simplied.katara;

public class PrintableError {
    private Integer row;
    private Integer column;
    private String value;

    public PrintableError(Integer row, Integer column, String value) {
        this.row = row;
        this.column = column;
        this.value = value;
    }

    public Integer getRow() {
        return row;
    }

    public void setRow(Integer row) {
        this.row = row;
    }

    public Integer getColumn() {
        return column;
    }

    public void setColumn(Integer column) {
        this.column = column;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "PrintableError{" +
                "row=" + row +
                ", column=" + column +
                ", value='" + value + '\'' +
                '}';
    }
}
