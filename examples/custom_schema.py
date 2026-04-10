"""
Example: Define a custom schema in code (no YAML) and run it.

Usage:
  python examples/custom_schema.py my_invoice.pdf
"""
import sys

from finlit import DocumentPipeline, Field, Schema

invoice_schema = Schema(
    name="custom_invoice",
    version="1.0",
    document_type="Vendor Invoice",
    fields=[
        Field(name="vendor_name", dtype=str, required=True,
              description="Name of the vendor"),
        Field(name="invoice_number", dtype=str, required=True,
              description="Invoice number as printed"),
        Field(name="invoice_date", dtype=str, required=True,
              description="Invoice date in YYYY-MM-DD"),
        Field(name="subtotal", dtype=float, required=True,
              description="Pre-tax subtotal"),
        Field(name="gst_hst", dtype=float, required=False,
              description="GST/HST charged"),
        Field(name="total", dtype=float, required=True,
              description="Total amount due"),
    ],
)

pipeline = DocumentPipeline(schema=invoice_schema, extractor="claude")
result = pipeline.run(sys.argv[1])

for field_name, value in result.fields.items():
    print(f"  {field_name:20s} {value}")
