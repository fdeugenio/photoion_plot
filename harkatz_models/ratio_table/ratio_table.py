from astropy import table

__all__ = ("ratio_table",)

# Ratty tables to calculate ratios on the fly.
class ratio_table(table.Table):
    """Silly class inheriting from `astropy.table.Table`. Issue is that inheriting
    from that class is a bit sticky, so I inherit, run the bit I need, then
    forego the inheritance (see `gutkintab.__class__ = table.table.Table` later on.)
    If you are reading this you may need a doctor.
    """
    def __getitem__(self, key):
        if type(key) is not str: # Remember how to slice tables
            return super(ratio_table, self).__getitem__(key)
        if key in self.colnames: # Return simple columns.
            return super(ratio_table, self).__getitem__(key)
        if '/' in key:
            return self.__getitem_div__(key)
        if '+' in key:
            return self.__getitem_plus__(key)

    def __getitem_plus__(self, key):
        try:
            keyadd1, keyadd2= key.split('+')
        except ValueError:
            message = f'Ambiguous {key}; column sums must have only one `+`'
            raise KeyError(message)
        else:
            return (
                super(ratio_table, self).__getitem__(keyadd1)
                + super(ratio_table, self).__getitem__(keyadd2)
            )

    def __getitem_div__(self, key):
        try:
            keynum, keyden = key.split('/')
        except ValueError:
            message = f'Ambiguous {key}; column ratios must have only one `/`'
            raise KeyError(message)
        else:
            return (
                super(ratio_table, self).__getitem__(keynum)
                / super(ratio_table, self).__getitem__(keyden)
            )
